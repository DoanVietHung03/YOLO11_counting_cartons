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
import queue
from collections import deque
from datetime import datetime
from mysql.connector.pooling import MySQLConnectionPool
from concurrent.futures import ThreadPoolExecutor

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

# ====================== CẤU HÌNH (CONFIG) ======================
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

CONFIG = {
    # Danh sách các luồng RTSP
    'RTSP_URLS': [
        'rtsp://rtsp-server:8554/mystream',
        'rtsp://rtsp-server:8554/mystream2'  # Luồng RTSP mới
    ],

    # IO / Model
    'MODEL_PATH': './best.pt',
    'CONF_THRESHOLD': 0.4,
    'CLASS_ID': 1,  # ID của lớp 'stamp'

    # Geometry / Preprocess
    'RESIZE_PERCENT': 60,

    # Counting Logic
    'MOVE_THRESHOLD': 5,
    'MIN_LIFETIME': 3,

    # Tracking
    'LOST_TRACK_BUFFER': 100,

    # Db batching
    'BATCH_SIZE': 6,

    # Frame scheduling (Adaptive Skip dựa trên thời gian)
    'FRAME_SKIP_MIN': 1, # Bỏ qua tối thiểu 1 frame (tức là xử lý gần như mọi frame)
    'FRAME_SKIP_MAX': 6, # Bỏ qua tối đa 6 frame khi hệ thống bị quá tải
    'LOAD_THRESHOLD_HIGH': 1.2, # Nếu (thời gian xử lý) > (thời gian 1 frame) * 1.2 -> Tăng skip
    'LOAD_THRESHOLD_LOW': 0.5, # Nếu (thời gian xử lý) < (thời gian 1 frame) * 0.5 -> Giảm skip

    # Frame buffer
    'USE_LATEST_ONLY': True,

    # Streaming
    'SEND_BINARY': True,
    'JPEG_QUALITY': 70,
}

# ===================== DB =========================
# Khởi tạo kết nối đến cơ sở dữ liệu
stream_ids = [i for i, _ in enumerate(CONFIG['RTSP_URLS'])]
initialize_database(stream_ids)

# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('rtsp-yolo-bytrack')

# ====================== DATABASE WORKER ======================
db_queue = queue.Queue()

def db_worker():
    while True:
        item = db_queue.get()
        if item is None:
            break
        logger.info(f"====== DB WORKER: Got item from queue. Preparing to insert. ======")
        stream_id, detections = item
        logger.info(f"Data for stream {stream_id}: {len(detections)} records.")
        try:
            db_insert(detections, stream_id)
            logger.info(f"Stream {stream_id}: Successfully inserted {len(detections)} records into DB.")
        except Exception as e:
            logger.error(f"Error inserting into DB for stream {stream_id}: {e}")
        db_queue.task_done()

threading.Thread(target=db_worker, daemon=True).start()

# ====================== VIDEO PROCESSOR CLASS ======================
class VideoProcessor:
    def __init__(self, config, stream_id):
        self.config = config
        self.stream_id = stream_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Stream {stream_id} using device: {self.device}')
        if self.device.type == 'cuda':
            logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")

        # Model
        self.model = self._load_model()
        self.class_names = self.model.model.names if hasattr(self.model.model, 'names') else getattr(self.model, 'names', {})

        # Tracker
        self.fps = self._probe_fps()
        logger.info(f'Stream {stream_id} probed FPS: {self.fps}')
        self.tracker = ByteTrack(frame_rate=int(self.fps), lost_track_buffer=self.config['LOST_TRACK_BUFFER'])

        # State Management
        self.track_memory = {}
        self.track_lock = threading.Lock()
        self.total_count = 0
        self.frame_idx = 0
        self.frame_skip = self.config['FRAME_SKIP_MIN']
        self.stamp_count = 0  # Đếm stamps để xóa frame cũ

        # Threading & Buffer
        self.frame_buffer = self._LatestFrame() if self.config['USE_LATEST_ONLY'] else deque(maxlen=10)
        self.stop_event = threading.Event()
        self.reader_thread = None
        self.web_status = False

    class _LatestFrame:
        def __init__(self):
            self._frame = None
            self._lock = threading.Lock()
        def put(self, frame):
            with self._lock: self._frame = frame
        def get(self):
            with self._lock: return None if self._frame is None else self._frame.copy()

    def _load_model(self):
        logger.info(f'Loading YOLO model for stream {self.stream_id}...')
        model = YOLO(self.config['MODEL_PATH']).to(self.device)
        try:
            model.fuse()
            logger.info(f'Model layers fused for stream {self.stream_id}.')
        except Exception: 
            logger.warning(f'Failed to fuse model layers for stream {self.stream_id}. Continuing without fusion.')
            pass
        if self.device.type == 'cuda':
            model.model.half()
            logger.info(f'FP16 inference enabled for stream {self.stream_id}.')
        return model

    def _probe_fps(self):
        cap = cv2.VideoCapture(self.config['RTSP_URLS'][self.stream_id], cv2.CAP_FFMPEG)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps and not math.isnan(fps) and fps > 0 else 25.0

    def _frame_reader_task(self):
        backoff = 2
        while not self.stop_event.is_set():
            if not self.web_status:
                time.sleep(1)
                continue

            cap = cv2.VideoCapture(self.config['RTSP_URLS'][self.stream_id], cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giới hạn buffer RTSP
            if not cap.isOpened():
                logger.error(f'Unable to open RTSP stream {self.stream_id}. Retrying in {backoff}s...')
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            
            backoff = 2
            logger.info(f"RTSP stream {self.stream_id} opened successfully.")
            while self.web_status and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.error(f'Stream {self.stream_id} disconnected. Reconnecting...')
                    break
                if self.config['RESIZE_PERCENT'] != 100:
                    h, w = frame.shape[:2]
                    nh, nw = int(h * self.config['RESIZE_PERCENT'] / 100), int(w * self.config['RESIZE_PERCENT'] / 100)
                    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                try:
                    self.frame_buffer.put(frame)
                except queue.Full:
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put(frame)
                    except queue.Empty:
                        logger.warning(f"Frame buffer full and empty for stream {self.stream_id}. Dropping frame.")
                        pass
                cv2.waitKey(1)
            cap.release()
        logger.info(f"Frame reader thread for stream {self.stream_id} stopped.")

    def start_reader(self):
        if self.reader_thread is None or not self.reader_thread.is_alive():
            self.stop_event.clear()
            self.reader_thread = threading.Thread(target=self._frame_reader_task, daemon=True)
            self.reader_thread.start()
            logger.info(f"Frame reader thread for stream {self.stream_id} started.")

    def stop_reader(self):
        self.stop_event.set()
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2)
        logger.info(f"Frame reader for stream {self.stream_id} stopped.")

    def set_web_status(self, status: bool):
        self.web_status = status
        logger.info(f"Web status for stream {self.stream_id} set to: {status}")
        if not status:
            self.stop_reader()
            with self.track_lock:
                self.track_memory.clear()
            self.total_count = 0
            self.frame_idx = 0
            self.stamp_count = 0
            if isinstance(self.frame_buffer, self._LatestFrame):
                self.frame_buffer._frame = None
            else:
                self.frame_buffer.clear()

    def process_and_draw_frame(self, frame: cv2.Mat):
        start_time = time.perf_counter()
        batch_detections = []
        
        with torch.no_grad():
            results = self.model(frame, verbose=False, conf=self.config['CONF_THRESHOLD'])[0]
        
        detections = Detections.from_ultralytics(results)
        tracked = self.tracker.update_with_detections(detections)
        
        h, w = frame.shape[:2]

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

            to_del = [tid for tid, m in self.track_memory.items() if self.frame_idx - m['last_seen'] > self.config['LOST_TRACK_BUFFER'] * 2]
            for tid in to_del:
                del self.track_memory[tid]

        cv2.line(frame, (w // 2, h // 6), (w // 2, h * 5 // 6), (0, 255, 255), 1)
        cv2.putText(frame, f'Stream {self.stream_id}: {self.total_count}', (int(w / 20), int(h / 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

        processing_time = time.perf_counter() - start_time
        return frame, batch_detections, processing_time

    def _update_track_memory(self, track_id, cx, cy, class_id, confidence, w, h, batch_detections):
        m = self.track_memory.get(track_id)
        if m is None:
            self.track_memory[track_id] = {'cx': cx, 'cy': cy, 'prev_cx': cx, 'prev_cy': cy, 'lifetime': 1, 'counted': False, 'passed_vertical': False, 'last_seen': self.frame_idx}
            return

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

        m['prev_cx'], m['prev_cy'] = m['cx'], m['cy']
        m['cx'], m['cy'] = cx, cy
        m['lifetime'] += 1
        m['last_seen'] = self.frame_idx
    
    def tune_frame_skip_by_time(self, processing_time: float):
        frame_interval = 1.0 / self.fps
        if processing_time > frame_interval * self.config['LOAD_THRESHOLD_HIGH']:
            self.frame_skip = min(self.frame_skip + 1, self.config['FRAME_SKIP_MAX'])
        elif processing_time < frame_interval * self.config['LOAD_THRESHOLD_LOW']:
            self.frame_skip = max(self.frame_skip - 1, self.config['FRAME_SKIP_MIN'])

# ====================== FASTAPI APP ======================
app = FastAPI()
import os
max_workers = max(4, min(os.cpu_count() or 4, len(CONFIG['RTSP_URLS']) * 2))
Executor = ThreadPoolExecutor(max_workers=max_workers)  # Tăng số worker để xử lý song song

# Khởi tạo các VideoProcessor cho mỗi luồng
video_processors = [
    VideoProcessor(CONFIG, i)
    for i, _ in enumerate(CONFIG['RTSP_URLS'])
]

@app.on_event("startup")
async def startup_event():
    for processor in video_processors:
        processor.start_reader()

@app.on_event("shutdown")
async def shutdown_event():
    for processor in video_processors:
        processor.stop_reader()
    db_queue.put(None)  # Dừng db_worker

# HTML giao diện với nhiều video
HTML = """
<!doctype html>
<html>
<head><meta charset='utf-8'><title>RTSP Streams</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;}
.video-container{display:flex;justify-content:center;flex-wrap:wrap;}
.video{display:block;margin:10px;width:auto;height:auto;object-fit:contain;max-width:none;border:1px solid #444;}
</style>
</head>
<body>
<h2 style='text-align:center'>Multi-RTSP → YOLOv11n + ByteTrack</h2>
<div class='video-container'>
%s
</div>
<script>
    const ws = new WebSocket('ws://' + location.host + '/ws');
    ws.binaryType = 'blob';
    ws.onmessage = (ev) => {
        if (ev.data instanceof Blob) {
            const reader = new FileReader();
            reader.onload = () => {
                try {
                    // Tìm vị trí của dấu phân cách "|"
                    const separatorIndex = reader.result.indexOf('|');
                    if (separatorIndex === -1) {
                        console.error('No separator found in WebSocket data');
                        return;
                    }
                    // Tách metadata và frame
                    const metadataStr = reader.result.slice(0, separatorIndex);
                    const frameData = ev.data.slice(separatorIndex + 1);
                    const data = JSON.parse(metadataStr);
                    const img = document.getElementById('video_stream_' + data.stream_id);
                    if (img) {
                        const url = URL.createObjectURL(frameData);
                        img.onload = () => URL.revokeObjectURL(url);
                        img.src = url;
                    } else {
                        console.error('Image element not found for stream_id: ' + data.stream_id);
                    }
                } catch (e) {
                    console.error('Error processing WebSocket data:', e);
                }
            };
            reader.readAsText(new Blob([ev.data.slice(0, 200)])); // Tăng kích thước để đảm bảo đọc hết metadata
        }
    };
    ws.onerror = (e) => console.error('WebSocket error:', e);
    ws.onclose = () => console.log('WebSocket closed');
</script>
</body>
</html>
""" % ''.join([f"<img id='video_stream_{i}' class='video' />" for i in range(len(CONFIG['RTSP_URLS']))])

@app.get('/')
async def index():
    return HTMLResponse(HTML)

@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    for processor in video_processors:
        processor.set_web_status(True)

    print(processor.stream_id)
    
    batch_to_db = {processor.stream_id: [] for processor in video_processors}
    try:
        async def process_stream(processor):
            while ws.client_state == WebSocketState.CONNECTED:
                frame = processor.frame_buffer.get()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                processor.frame_idx += 1
                processor.stamp_count += 1

                processed_frame = None
                if processor.frame_idx % processor.frame_skip == 0:
                    processed_frame, new_detections, processing_time = await run_in_threadpool(
                        processor.process_and_draw_frame, frame
                    )
                    processor.tune_frame_skip_by_time(processing_time)
                    if new_detections:
                        batch_to_db[processor.stream_id].extend(new_detections)
                else:
                    processed_frame, _, _ = await run_in_threadpool(
                        processor.process_and_draw_frame, frame
                    )

                # Kiểm tra và gửi batch đến DB
                logger.info(f"Stream {processor.stream_id}: Current batch size: {len(batch_to_db[processor.stream_id])}")
                if batch_to_db[processor.stream_id] and len(batch_to_db[processor.stream_id]) >= processor.config['BATCH_SIZE']:
                    db_queue.put((processor.stream_id, batch_to_db[processor.stream_id].copy()))
                    batch_to_db[processor.stream_id].clear()
                    if processor.stamp_count >= 6:
                        if isinstance(processor.frame_buffer, processor._LatestFrame):
                            processor.frame_buffer._frame = None
                        else:
                            processor.frame_buffer.clear()
                        processor.stamp_count = 0
                        logger.info(f"Cleared old frames for stream {processor.stream_id} after 6 stamps.")

                if processed_frame is not None:
                    logger.debug(f"Processed frame shape: {processed_frame.shape}")
                    ok, buf = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG['JPEG_QUALITY']])
                    if ok:
                        metadata = json.dumps({"stream_id": processor.stream_id}).encode()
                        logger.debug(f"Sending frame for stream {processor.stream_id}, metadata size: {len(metadata)}, frame size: {len(buf)}")
                        await ws.send_bytes(metadata + b"|" + buf.tobytes())
                    else:
                        logger.error(f"Failed to encode frame for stream {processor.stream_id}")

                await asyncio.sleep(0)

        await asyncio.gather(*(process_stream(processor) for processor in video_processors))

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f'WebSocket error: {e}')
    finally:
        for processor in video_processors:
            processor.set_web_status(False)
        for stream_id in batch_to_db:
            batch_to_db[stream_id].clear()
        logger.info("WebSocket connection closed.")

if __name__ == '__main__':
    import uvicorn
    import json
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)