import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import time
import math
import base64
import torch
import logging
import asyncio
import threading
from collections import deque
from datetime import datetime

# FastAPI và các thành phần liên quan
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketState

# Các thư viện xử lý ảnh và AI
from ultralytics import YOLO
from supervision import ByteTrack, Detections

# Module tự định nghĩa
from mySQL_db import db_insert, initialize_database

initialize_database()

# ====================== CẤU HÌNH (CONFIG) ======================
# Gộp tất cả cấu hình vào một nơi để dễ dàng thay đổi
CONFIG = {
    # IO / Model
    'RTSP_URL': 'rtsp://rtsp-server:8554/mystream',
    'MODEL_PATH': './best.pt',
    'CONF_THRESHOLD': 0.4,
    'CLASS_ID': 1,  # ID của lớp 'stamp'

    # Geometry / Preprocess
    'RESIZE_PERCENT': 60,  # Giảm kích thước khung hình để xử lý nhanh hơn

    # Counting Logic
    'MOVE_THRESHOLD': 5,
    'MIN_LIFETIME': 3,

    # Tracking
    'LOST_TRACK_BUFFER': 100,

    # DB batching
    'BATCH_SIZE': 6,

    # Frame scheduling (Adaptive Skip dựa trên thời gian)
    'FRAME_SKIP_MIN': 1,      # Bỏ qua tối thiểu 1 frame (tức là xử lý gần như mọi frame)
    'FRAME_SKIP_MAX': 6,      # Bỏ qua tối đa 6 frame khi hệ thống bị quá tải
    'LOAD_THRESHOLD_HIGH': 1.2, # Nếu (thời gian xử lý) > (thời gian 1 frame) * 1.2 -> Tăng skip
    'LOAD_THRESHOLD_LOW': 0.5,  # Nếu (thời gian xử lý) < (thời gian 1 frame) * 0.5 -> Giảm skip

    # Frame buffer
    'USE_LATEST_ONLY': True,  # Chỉ xử lý frame mới nhất để giảm độ trễ

    # Streaming
    'SEND_BINARY': True,
    'JPEG_QUALITY': 70,
}

# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('rtsp-yolo-bytrack')

# =================================================================================
# LỚP XỬ LÝ VIDEO TRUNG TÂM (VIDEO PROCESSOR CLASS)
# =================================================================================
class VideoProcessor:
    """
    Gói gọn tất cả logic vào một class để dễ quản lý.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        if self.device.type == 'cuda':
            logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")

        # --- Model ---
        self.model = self._load_model()
        self.class_names = self.model.model.names if hasattr(self.model.model, 'names') else getattr(self.model, 'names', {})

        # --- Tracker ---
        self.fps = self._probe_fps()
        logger.info(f'Probed FPS: {self.fps}')
        self.tracker = ByteTrack(frame_rate=int(self.fps), lost_track_buffer=self.config['LOST_TRACK_BUFFER'])

        # --- State Management ---
        self.track_memory = {}
        self.track_lock = threading.Lock()
        self.total_count = 0
        self.frame_idx = 0
        self.frame_skip = self.config['FRAME_SKIP_MIN']

        # --- Threading & Buffer ---
        self.frame_buffer = self._LatestFrame() if self.config['USE_LATEST_ONLY'] else deque(maxlen=10)
        self.stop_event = threading.Event()
        self.reader_thread = None
        self.web_status = False # Trạng thái kết nối của client

    class _LatestFrame:
        """ Giữ frame mới nhất trong bộ đệm, an toàn cho đa luồng."""
        def __init__(self):
            self._frame = None
            self._lock = threading.Lock()
        def put(self, frame):
            with self._lock: self._frame = frame
        def get(self):
            with self._lock: return None if self._frame is None else self._frame.copy()

    def _load_model(self):
        logger.info('Loading YOLO model...')
        model = YOLO(self.config['MODEL_PATH']).to(self.device)
        try:
            model.fuse()
            logger.info('Model layers fused for faster inference.')
        except Exception: pass
        if self.device.type == 'cuda':
            model.model.half()
            logger.info('FP16 inference enabled.')
        logger.info('Model loaded.')
        return model

    def _probe_fps(self):
        cap = cv2.VideoCapture(self.config['RTSP_URL'], cv2.CAP_FFMPEG)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps and not math.isnan(fps) and fps > 0 else 25.0

    def _frame_reader_task(self):
        """Task for reading frames from RTSP stream in a background thread."""
        backoff = 2
        while not self.stop_event.is_set():
            if not self.web_status:
                time.sleep(1) # Tạm nghỉ khi không có client kết nối
                continue

            cap = cv2.VideoCapture(self.config['RTSP_URL'], cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logger.error(f'Unable to open RTSP stream. Retrying in {backoff}s...')
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            
            backoff = 2
            logger.info("RTSP stream opened successfully.")
            while self.web_status and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.error('Stream disconnected. Reconnecting...')
                    break

                # Xử lý trước khi đưa vào bộ đệm
                if self.config['RESIZE_PERCENT'] != 100:
                    h, w = frame.shape[:2]
                    nh, nw = int(h * self.config['RESIZE_PERCENT'] / 100), int(w * self.config['RESIZE_PERCENT'] / 100)
                    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                self.frame_buffer.put(frame)
            cap.release()
        logger.info("Frame reader thread stopped.")

    def start_reader(self):
        """Bắt đầu đọc frame từ RTSP stream trong một luồng riêng."""
        if self.reader_thread is None or not self.reader_thread.is_alive():
            self.stop_event.clear()
            self.reader_thread = threading.Thread(target=self._frame_reader_task, daemon=True)
            self.reader_thread.start()
            logger.info("Frame reader thread started.")

    def stop_reader(self):
        """Dừng đọc frame từ RTSP stream."""
        self.stop_event.set()
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2)
        logger.info("Frame reader stop signal sent.")

    def set_web_status(self, status: bool):
        self.web_status = status
        logger.info(f"Web status set to: {status}")
        if not status:
            self.stop_reader()

            # Reset state khi không còn client kết nối
            with self.track_lock:
                self.track_memory.clear()
            self.total_count = 0
            self.frame_idx = 0

            # Xóa frame trong buffer
            if isinstance(self.frame_buffer, self._LatestFrame):
                self.frame_buffer._frame = None
            else:
                self.frame_buffer.clear()


    def process_and_draw_frame(self, frame: cv2.Mat):
        """
        Đây là hàm CPU-bound, sẽ được chạy trong threadpool.
        """
        start_time = time.perf_counter()

        batch_detections = []
        
        # --- Inference ---
        with torch.no_grad():
            results = self.model(frame, verbose=False, conf=self.config['CONF_THRESHOLD'])[0]
        
        detections = Detections.from_ultralytics(results)
        tracked = self.tracker.update_with_detections(detections)
        
        h, w = frame.shape[:2]
        
        # --- Cập nhật tracks và count ---
        with self.track_lock:
            for i in range(len(tracked)):
                x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                track_id = int(tracked.tracker_id[i])
                class_id = int(tracked.class_id[i])
                confidence = float(tracked.confidence[i])

                if class_id != self.config['CLASS_ID']:
                    continue
                
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                self._update_track_memory(track_id, cx, cy, class_id, confidence, w, h, batch_detections)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

            # --- Dọn dẹp tracks cũ ---
            to_del = [tid for tid, m in self.track_memory.items() if self.frame_idx - m['last_seen'] > self.config['LOST_TRACK_BUFFER'] * 2]
            for tid in to_del:
                del self.track_memory[tid]

        # --- Draw visuals ---
        cv2.line(frame, (w // 2, h // 6), (w // 2, h * 5 // 6), (0, 255, 255), 1)
        cv2.putText(frame, f'Total: {self.total_count}', (int(w / 20), int(h / 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

        processing_time = time.perf_counter() - start_time # Kết thúc đo
        
        return frame, batch_detections, processing_time

    def _update_track_memory(self, track_id, cx, cy, class_id, confidence, w, h, batch_detections):
        m = self.track_memory.get(track_id)
        if m is None:
            self.track_memory[track_id] = {'cx': cx, 'cy': cy, 'prev_cx': cx, 'prev_cy': cy, 'lifetime': 1, 'counted': False, 'passed_vertical': False, 'last_seen': self.frame_idx}
            return

        # kiểm tra điều kiện đếm
        moved = abs(m['prev_cx'] - cx) >= self.config['MOVE_THRESHOLD'] or abs(m['prev_cy'] - cy) >= self.config['MOVE_THRESHOLD']
        mid_x = w // 2
        crossed = (m['prev_cx'] < mid_x <= cx) or (m['prev_cx'] >= mid_x > cx)
        in_range = (h * 1/6) <= cy <= (h * 5/6)

        if m['lifetime'] >= self.config['MIN_LIFETIME'] and not m['counted'] and moved and crossed and in_range and not m['passed_vertical']:
            self.total_count += 1
            m['counted'] = True
            m['passed_vertical'] = True
            batch_detections.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'track_id': track_id,
                'label': self.class_names.get(class_id, str(class_id)),
                'confidence': round(confidence, 3),
            })

        # Cập nhật state
        m['prev_cx'], m['prev_cy'] = m['cx'], m['cy']
        m['cx'], m['cy'] = cx, cy
        m['lifetime'] += 1
        m['last_seen'] = self.frame_idx
    
    def tune_frame_skip_by_time(self, processing_time: float):
        """
        Tối ưu quan trọng: Điều chỉnh frame_skip dựa trên thời gian xử lý.
        Nếu xử lý quá chậm -> tăng skip. Nếu quá nhanh -> giảm skip.
        """
        frame_interval = 1.0 / self.fps
        if processing_time > frame_interval * self.config['LOAD_THRESHOLD_HIGH']:
            self.frame_skip = min(self.frame_skip + 1, self.config['FRAME_SKIP_MAX'])
        elif processing_time < frame_interval * self.config['LOAD_THRESHOLD_LOW']:
            self.frame_skip = max(self.frame_skip - 1, self.config['FRAME_SKIP_MIN'])

# ====================== FASTAPI APP ======================
app = FastAPI()
video_processor = VideoProcessor(CONFIG)

@app.on_event("startup")
async def startup_event():
    video_processor.start_reader()

@app.on_event("shutdown")
async def shutdown_event():
    video_processor.stop_reader()

HTML = """
<!doctype html>
<html>
<head><meta charset='utf-8'><title>RTSP Stream</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;} #video{display:block;margin:20px auto;width:80%;max-width:960px;border:1px solid #444;}</style>
</head>
<body>
<h2 style='text-align:center'>RTSP → YOLOv11n + ByteTrack (Optimized)</h2>
<img id='video' />
<script>
    const ws=new WebSocket('ws://'+location.host+'/ws');
    const img=document.getElementById('video');
    ws.binaryType='blob';
    ws.onmessage=(ev)=>{if(ev.data instanceof Blob){const url=URL.createObjectURL(ev.data);img.onload=()=>URL.revokeObjectURL(url);img.src=url;}else{img.src='data:image/jpeg;base64,'+ev.data;}};
</script>
</body></html>
"""

@app.get('/')
async def index():
    return HTMLResponse(HTML)

@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    video_processor.set_web_status(True)
    
    batch_to_db = []
    try:
        while ws.client_state == WebSocketState.CONNECTED:
            frame = video_processor.frame_buffer.get()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            video_processor.frame_idx += 1

            processed_frame = None
            
            # --- Xử lý frame ---
            if video_processor.frame_idx % video_processor.frame_skip == 0:    
                # Chạy tác vụ nặng trong threadpool và nhận lại kết quả
                processed_frame, new_detections, processing_time = await run_in_threadpool(
                    video_processor.process_and_draw_frame, frame
                )
                
                # Điều chỉnh frame_skip với thời gian chính xác
                video_processor.tune_frame_skip_by_time(processing_time)
                
                if new_detections:
                    batch_to_db.extend(new_detections)
            else:
                # QUAN TRỌNG: Kể cả khi skip, vẫn cần vẽ HUD mà không block server
                # Chúng ta vẫn dùng threadpool nhưng không cần quan tâm đến kết quả detections
                processed_frame, _, _ = await run_in_threadpool(
                    video_processor.process_and_draw_frame, frame
                )

            # --- Lưu vào DB ---
            if batch_to_db and len(batch_to_db) >= video_processor.config['BATCH_SIZE']:
                await run_in_threadpool(db_insert, batch_to_db.copy())
                batch_to_db.clear()

            # --- Gửi frame cho client ---
            if processed_frame is not None:
                ok, buf = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG['JPEG_QUALITY']])
                if ok:
                    # Gửi đi frame đã được xử lý (hoặc chỉ được vẽ HUD)
                    await ws.send_bytes(buf.tobytes())
            
            await asyncio.sleep(0)

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f'WebSocket error: {e}')
    finally:
        # Khi client ngắt kết nối, chỉ cần cập nhật trạng thái
        # Không dừng server, sẵn sàng cho client mới
        video_processor.set_web_status(False)
        batch_to_db.clear()
        logger.info("WebSocket connection closed.")

if __name__ == '__main__':
    import uvicorn
    # Chạy server với 1 worker duy nhất để đảm bảo chỉ có 1 instance VideoProcessor
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)