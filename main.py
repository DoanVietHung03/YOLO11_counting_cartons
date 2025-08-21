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
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool
from ultralytics import YOLO
from supervision import ByteTrack, Detections
from mySQL_db import db_insert
import config

# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('rtsp-yolo-bytrack')

# ====================== TORCH / CUDA ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')
if device.type == 'cuda':
    logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")

torch.backends.cudnn.benchmark = True  # Optimize for varying input sizes

#====================== OPENCV RTSP (TCP) ======================#
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

#====================== CONFIG ======================#
CONFIG = {
    # IO / Model
    'RTSP_URL': 'rtsp://rtsp-server:8554/mystream',
    'MODEL_PATH': './best.pt',
    'CONF_THRESHOLD': 0.4,
    'CLASS_ID': 1,  # stamp

    # Geometry / Preprocess
    'RESIZE_PERCENT': None,   # downscale to save compute

    # Counting Logic
    'MOVE_THRESHOLD': 5,
    'MIN_LIFETIME': 3,

    # Tracking
    'LOST_TRACK_BUFFER': 100,

    # DB batching and status
    'BATCH_SIZE': 6,

    # Frame scheduling
    'FRAME_SKIP_MIN': 1,
    'FRAME_SKIP_MAX': 6,
    'ADAPTIVE_SKIP': True,            # auto tune skip by backlog
    'QUEUE_HIGH_WATERMARK': 5,        # when backlog >= -> increase skip
    'QUEUE_LOW_WATERMARK': 1,         # when backlog <= -> decrease skip

    # Frame buffer
    'USE_LATEST_ONLY': True,          # ultra-low latency: overwrite single-slot buffer
    'QUEUE_SIZE': 540,                # used when USE_LATEST_ONLY=False

    # Streaming
    'SEND_BINARY': True,              # WebSocket send bytes (faster than base64)
    'JPEG_QUALITY': 80,
}

rtsp_last = CONFIG['RTSP_URL'].rstrip('/').split('/')[-1]
CONFIG['RESIZE_PERCENT'] = 40 if rtsp_last == 'mystream2' else 60

# ====================== GLOBAL STATE ======================
TOTAL_COUNT = 0
FRAME_IDX = 0
stop_event = asyncio.Event()

# ====================== MODEL ======================
logger.info('Loading YOLO model...')
model = YOLO(CONFIG['MODEL_PATH']).to(device)

# Fuse conv+bn for speed (if supported)
try:
    model.fuse()  # type: ignore
    logger.info('Model layers fused for faster inference.')
except Exception as e:
    logger.info(f'Fuse not available: {e}')

# Enable half precision on CUDA if possible
if device.type == 'cuda':
    try:
        model.model.half()
        logger.info('FP16 inference enabled.')
    except Exception as e:
        logger.warning(f'FP16 not supported: {e}')

# torch.compile (PyTorch >= 2.0)
try:
    model.model = torch.compile(model.model)  # type: ignore[attr-defined]
    logger.info('Model compiled with torch.compile().')
except Exception as e:
    logger.info(f'torch.compile unavailable or failed: {e}')

CLASS_NAMES = model.model.names if hasattr(model.model, 'names') else getattr(model, 'names', {})

# ====================== FPS / TRACKER ======================

def probe_fps(rtsp_url: str) -> float:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 25.0
    return fps

FPS = probe_fps(CONFIG['RTSP_URL'])
logger.info(f'Using FPS: {FPS}')
tracker = ByteTrack(frame_rate=int(FPS), lost_track_buffer=CONFIG['LOST_TRACK_BUFFER'])

# ====================== TRACK MEMORY ======================
track_memory = {}
track_lock = threading.Lock()

# ====================== FRAME BUFFER ======================
class LatestFrame:
    """A single-slot buffer for the most recent frame to minimize latency."""
    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def put(self, frame):
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

if CONFIG['USE_LATEST_ONLY']:
    frame_buffer = LatestFrame()
else:
    from queue import Queue
    frame_buffer = Queue(maxsize=CONFIG['QUEUE_SIZE'])

# ====================== FRAME READER THREAD ======================

def preprocess_frame(frame):
    if CONFIG['RESIZE_PERCENT'] and CONFIG['RESIZE_PERCENT'] != 100:
        h, w = frame.shape[:2]
        nh = int(h * CONFIG['RESIZE_PERCENT'] // 100)
        nw = int(w * CONFIG['RESIZE_PERCENT'] // 100)
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return frame

def handle_frame(frame):
    frame = preprocess_frame(frame)
    if isinstance(frame_buffer, LatestFrame):
        frame_buffer.put(frame)
    else:
        # Drop oldest if full to avoid lag
        if frame_buffer.full():
            try:
                frame_buffer.get_nowait()
            except Exception:
                pass
        frame_buffer.put(frame)

def handle_stream(cap):
    while config.WEB_STATUS:
        ret, frame = cap.read()
        if not ret:
            logger.error('Stream disconnected. Reconnecting...')
            return False
        handle_frame(frame)
    return True

def frame_reader(rtsp_url: str):
    backoff = 2
    while not stop_event.is_set():
        if not config.WEB_STATUS:
            time.sleep(1)
            continue

        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error('Unable to open RTSP stream. Retrying...')
            time.sleep(backoff)
            backoff = min(backoff * 2, 10)
            continue

        backoff = 2
        if not handle_stream(cap):
            cap.release()
            continue
        cap.release()

reader_thread = threading.Thread(target=frame_reader, args=(CONFIG['RTSP_URL'],), daemon=True)
reader_thread.start()

# ====================== COUNTING / TRACK HELPERS ======================

def update_track_memory(track_id, cx, cy, class_id, confidence, frame_idx, w, h, batch_detections):
    global TOTAL_COUNT
    m = track_memory.get(track_id)
    if m is None:
        track_memory[track_id] = {
            'cx': cx, 'cy': cy,
            'prev_cx': cx, 'prev_cy': cy,
            'lifetime': 1,
            'counted': False,
            'passed_vertical': False,
            'last_seen': frame_idx,
        }
        return batch_detections

    prev_cx, prev_cy = m['prev_cx'], m['prev_cy']
    lifetime = m['lifetime']

    moved_enough = abs(prev_cx - cx) >= CONFIG['MOVE_THRESHOLD'] or abs(prev_cy - cy) >= CONFIG['MOVE_THRESHOLD']
    mid_x = int(w / 2 + 5)
    crossed_vertical = (prev_cx < mid_x <= cx) or (prev_cx >= mid_x > cx)
    in_vertical_y_range = int(h * 1/6) <= cy <= int(h * 5/6)

    if lifetime >= CONFIG['MIN_LIFETIME'] and (not m['counted']) and moved_enough and crossed_vertical and in_vertical_y_range and (not m['passed_vertical']):
        TOTAL_COUNT += 1
        m['counted'] = True
        m['passed_vertical'] = True

        frame_det = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'track_id': track_id,
            'label': CLASS_NAMES.get(class_id, str(class_id)),
            'confidence': round(confidence, 3),
            'bbox': [int(prev_cx), int(prev_cy), int(cx), int(cy)],
        }
        batch_detections.append(frame_det)

    m['prev_cx'], m['prev_cy'] = m['cx'], m['cy']
    m['cx'], m['cy'] = cx, cy
    m['lifetime'] += 1
    m['last_seen'] = frame_idx
    return batch_detections


def process_frame(frame, frame_idx, batch_detections):
    # NOTE: Ultralytics handles dtype internally; we pass uint8 BGR frame
    with torch.no_grad():
        results = model(frame, verbose=False, conf=CONFIG['CONF_THRESHOLD'])[0]

    detections = Detections.from_ultralytics(results)
    tracked = tracker.update_with_detections(detections)

    h, w = frame.shape[:2]

    for i in range(len(tracked)):
        x1, y1, x2, y2 = map(int, tracked.xyxy[i])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        track_id = int(tracked.tracker_id[i])
        class_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
        confidence = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

        # gate by class & conf
        if class_id != CONFIG['CLASS_ID'] or confidence < CONFIG['CONF_THRESHOLD']:
            tm = track_memory.get(track_id)
            if tm:
                tm['last_seen'] = frame_idx
            continue

        update_track_memory(track_id, cx, cy, class_id, confidence, frame_idx, w, h, batch_detections)

        # draw box (keep simple to save CPU)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

    return frame, batch_detections


def draw_visuals(frame):
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, h // 6), (w // 2, h * 5 // 6), (0, 255, 255), 1)
    cv2.putText(frame, f'Total: {TOTAL_COUNT}', (int(w // 20), int(h // 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
    return frame


# ====================== ADAPTIVE SKIP ======================
frame_skip = CONFIG['FRAME_SKIP_MIN']


def tune_frame_skip(backlog: int):
    global frame_skip
    if not CONFIG['ADAPTIVE_SKIP']:
        return
    if backlog >= CONFIG['QUEUE_HIGH_WATERMARK']:
        frame_skip = min(frame_skip + 1, CONFIG['FRAME_SKIP_MAX'])
    elif backlog <= CONFIG['QUEUE_LOW_WATERMARK']:
        frame_skip = max(frame_skip - 1, CONFIG['FRAME_SKIP_MIN'])


# ====================== FASTAPI APP ======================
app = FastAPI()

HTML = f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>RTSP Stream</title>
  <style>
    body {{ background:#111; color:#eee; font-family:sans-serif; }}
    #video {{ display:block; margin: 20px auto; width:80%; max-width:960px; border:1px solid #444; }}
  </style>
</head>
<body>
  <h2 style='text-align:center'>RTSP → YOLOv11 + ByteTrack (binary WS)</h2>
  <img id='video' />
  <script>
    const ws = new WebSocket('ws://'+location.host+'/ws');
    const img = document.getElementById('video');

    ws.binaryType = 'blob';
    ws.onmessage = (ev) => {{
      // ev.data is Blob when server sends binary
      if (ev.data instanceof Blob) {{
        const url = URL.createObjectURL(ev.data);
        img.onload = () => URL.revokeObjectURL(url);
        img.src = url;
      }} else {{
        // text fallback: base64-encoded JPEG
        img.src = 'data:image/jpeg;base64,' + ev.data;
      }}
    }}
  </script>
</body>
</html>
"""

@app.get('/')
async def index():
    return HTMLResponse(HTML)

@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    global FRAME_IDX, TOTAL_COUNT
    config.WEB_STATUS = True
    logger.info(f'Client connected. WEB_STATUS = {config.WEB_STATUS}')

    batch_detections = []
    show_visuals = True

    disconnected = asyncio.Event()

    async def monitor_disconnect():
        try:
            while True:
                await ws.receive()  # Waits for any client message (none expected) or disconnect
        except WebSocketDisconnect:
            disconnected.set()
        except Exception as e:
            logger.error(f'Receive error: {e}')
            disconnected.set()

    monitor_task = asyncio.create_task(monitor_disconnect())

    try:
        while not stop_event.is_set():
            if not config.WEB_STATUS or disconnected.is_set():
                break

            # Pull latest frame
            if isinstance(frame_buffer, LatestFrame):
                frame = frame_buffer.get()
                backlog = 0
            else:
                try:
                    frame = frame_buffer.get(timeout=0.2)
                except Exception:
                    frame = None
                backlog = frame_buffer.qsize() if frame is not None else 0

            if frame is None:
                await asyncio.sleep(0.01)
                continue

            FRAME_IDX += 1
            # Adaptive frame skip based on backlog
            tune_frame_skip(backlog)

            # Run detection only on every Nth frame
            if FRAME_IDX % frame_skip == 0:
                frame, batch_detections = process_frame(frame, FRAME_IDX, batch_detections)

                # Clean up old tracks
                with track_lock:
                    to_del = [tid for tid, m in track_memory.items() if FRAME_IDX - m.get('last_seen', FRAME_IDX) > CONFIG['LOST_TRACK_BUFFER'] * 2]
                    for tid in to_del:
                        del track_memory[tid]

            # Draw HUD
            if show_visuals:
                frame = draw_visuals(frame)

            if config.WEB_STATUS and batch_detections and (len(batch_detections) >= CONFIG['BATCH_SIZE'] or FRAME_IDX % 100 == 0):
                await run_in_threadpool(db_insert, batch_detections)
                batch_detections = []

            # Encode JPEG once per frame sent
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG['JPEG_QUALITY']]
            ok, buf = cv2.imencode('.jpg', frame, encode_params)
            if not ok:
                continue

            if CONFIG['SEND_BINARY']:
                await ws.send_bytes(buf.tobytes())
            else:
                b64 = base64.b64encode(buf).decode('utf-8')
                await ws.send_text(b64)

            # Cooperative yield to event loop
            await asyncio.sleep(0)

    except Exception as e:
        logger.error(f'WebSocket error: {e}')
    finally:
        stop_event.set()
        config.WEB_STATUS = False
        batch_detections = []
        logger.info(f'Client disconnected. WEB_STATUS = {config.WEB_STATUS}')
        if not isinstance(frame_buffer, LatestFrame):
            while not frame_buffer.empty():
                frame_buffer.get_nowait()
        track_memory.clear()
        await ws.close()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)