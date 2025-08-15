import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from ultralytics import YOLO  # Detection Model
from supervision import ByteTrack, Detections  # Tracking method
from datetime import datetime
import time
import threading
import queue
from mySQL_db import db_insert  # Save data to db
import logging
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#========USING GPU IF AVAILABLE=======#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#========SET UP ENV FOR RTSP STREAM (USING TCP)==========#
import os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

#========CONFIG VARIABLES==========#
CONFIG = {
    "RTSP_URL": "rtsp://localhost:46458/mystream2",
    "MODEL_PATH": "D:/Test/best.pt",
    "CONF_THRESHOLD": 0.4,
    "MOVE_THRESHOLD": 5,
    "MIN_LIFETIME": 3,
    "QUEUE_SIZE": 640,
    "LOST_TRACK_BUFFER": 60,
    "BATCH_SIZE": 20,  # For database insertions
    "FRAME_SKIP": 1,  # Process every Nth frame (1 = no skip)
    "RESIZE_PERCENT": 30,
    "CLASS_ID": 1  # Assuming stamp class_id = 1
}

total_count = 0  # Total count of detected objects

#========MODEL LOADING========#
best_model = YOLO(CONFIG["MODEL_PATH"]).to(device)
try:
    best_model.model.half()  # Enable FP16 inference if supported
except Exception as e:
    logger.warning(f"Mixed precision not supported on this device: {e}")

#========STREAM READING FOR FPS========#
cap = cv2.VideoCapture(CONFIG["RTSP_URL"], cv2.CAP_FFMPEG)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()  # Release this cap as we'll use threaded reader

#========TRACKING (USING BYTETRACK)========#
tracker = ByteTrack(frame_rate=int(fps), lost_track_buffer=CONFIG["LOST_TRACK_BUFFER"])
track_memory = {}  # Store information for each track_id
frame_idx = 0

#=======FRAME READING USING THREAD========#
frame_queue = queue.Queue(maxsize=CONFIG["QUEUE_SIZE"])

# Thread to read frames from the RTSP stream with reconnection
def frame_reader(rtsp_url, q):
    while True:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error("Unable to open stream, retrying in 5 seconds...")
            time.sleep(5)
            continue
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Stream disconnected, reconnecting...")
                break
            # Resize in the reader thread
            h, w = frame.shape[:2]
            # The lines `h = h * CONFIG["RESIZE_PERCENT"] // 100` and `w = w *
            # CONFIG["RESIZE_PERCENT"] // 100` are resizing the height (`h`) and width (`w`) of the
            # frame by a certain percentage specified in the `CONFIG` dictionary.
            h = h * CONFIG["RESIZE_PERCENT"] // 100
            w = w * CONFIG["RESIZE_PERCENT"] // 100
            # frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if q.full():
                logger.warning("Frame queue is full, skipping oldest frame to avoid lag.")
                q.get()  # Remove oldest frame
            q.put(frame)
        cap.release()

# Start the frame reader thread
reader_thread = threading.Thread(target=frame_reader, args=(CONFIG["RTSP_URL"], frame_queue))
reader_thread.daemon = True
reader_thread.start()

# Modular function for processing detections and tracking
def update_track_memory(track_id, cx, cy, class_id, confidence, frame_idx, w, h, track_memory, batch_detections, results):
    global total_count
    if track_id not in track_memory:
        track_memory[track_id] = {
            "cx": cx, "cy": cy,
            "prev_cx": cx, "prev_cy": cy,
            "lifetime": 1,
            "counted": False,
            "passed_vertical": False,
            'last_seen': frame_idx
        }
    else:
        prev_cx = track_memory[track_id]["prev_cx"]
        prev_cy = track_memory[track_id]["prev_cy"]
        lifetime = track_memory[track_id]["lifetime"]

        moved_enough = abs(prev_cx - cx) >= CONFIG["MOVE_THRESHOLD"] or abs(prev_cy - cy) >= CONFIG["MOVE_THRESHOLD"]
        crossed_vertical = (prev_cx < int(w/2 + 5) and cx >= int(w/2 + 5) or prev_cx >= int(w/2 + 5) and cx < int(w/2 + 5))
        in_vertical_y_range = (cy >= int(h * 1/6) and cy <= int(h * 5/6))

        if lifetime >= CONFIG["MIN_LIFETIME"] and not track_memory[track_id]["counted"] and moved_enough and crossed_vertical and in_vertical_y_range and not track_memory[track_id]["passed_vertical"]:
            total_count += 1
            track_memory[track_id]["counted"] = True
            track_memory[track_id]["passed_vertical"] = True

            frame_detections = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "track_id": track_id,
                "label": results.names[class_id],
                "confidence": round(confidence, 3),
                "bbox": [int(prev_cx), int(prev_cy), int(cx), int(cy)]
            }
            batch_detections.append(frame_detections)

        track_memory[track_id]["prev_cx"] = track_memory[track_id]["cx"]
        track_memory[track_id]["prev_cy"] = track_memory[track_id]["cy"]
        track_memory[track_id]["cx"] = cx
        track_memory[track_id]["cy"] = cy
        track_memory[track_id]["lifetime"] += 1
        track_memory[track_id]["last_seen"] = frame_idx
    return batch_detections

def process_frame(frame, model, tracker, track_memory, frame_idx, w, h, batch_detections):
    with torch.no_grad():
        results = model(frame, verbose=False, conf=0.6)[0]
    detections = Detections.from_ultralytics(results)
    tracked = tracker.update_with_detections(detections)
    current_ids = set()

    for i in range(len(tracked)):
        x1, y1, x2, y2 = map(int, tracked.xyxy[i])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        track_id = int(tracked.tracker_id[i])
        class_id = int(tracked.class_id[i])
        confidence = float(tracked.confidence[i])

        current_ids.add(track_id)

        if class_id != CONFIG["CLASS_ID"] or confidence < CONFIG["CONF_THRESHOLD"]:
            if track_id in track_memory:
                track_memory[track_id]['last_seen'] = frame_idx
            continue

        batch_detections = update_track_memory(
            track_id, cx, cy, class_id, confidence, frame_idx, w, h, track_memory, batch_detections, results
        )

        label = f"{results.names[class_id]} {track_id}: {confidence:.2f}"
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return tracked, current_ids, batch_detections

# ========= FASTAPI APP =========
app = FastAPI()

# HTML to serve the WebSocket client
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Stream</title>
</head>
<body>
    <h1>RTSP Video Stream</h1>
    <img id="video" src="">
    <script>
        const ws = new WebSocket("ws://localhost:8000/ws");
        ws.onmessage = function(event) {
            const video = document.getElementById("video");
            video.src = "data:image/jpeg;base64," + event.data;
        };
        ws.onclose = function() {
            console.log("WebSocket connection closed");
        };
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    batch_detections = []
    show_visuals = True
    global total_count, frame_idx, track_memory
    try:
        while True:
            if not frame_queue.empty():
                await process_and_send_frame(
                    websocket, batch_detections, show_visuals
                )
                await asyncio.sleep(0.01)  # Prevent blocking the event loop
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

async def process_and_send_frame(websocket, batch_detections, show_visuals):
    global total_count, frame_idx, track_memory
    frame = frame_queue.get()
    h, w = frame.shape[:2]
    frame_idx += 1

    # Frame skipping for detection
    if frame_idx % CONFIG["FRAME_SKIP"] == 0:
        tracked, current_ids, batch_detections = process_frame(
            frame, best_model, tracker, track_memory, frame_idx, w, h, batch_detections
        )
    else:
        # On skipped frames, update tracker with empty detections
        detections = Detections.empty()
        tracked = tracker.update_with_detections(detections)
        current_ids = set()

    # Clean up old tracks
    track_memory = {
        tid: m
        for tid, m in track_memory.items()
        if frame_idx - m.get("last_seen", frame_idx) <= CONFIG["LOST_TRACK_BUFFER"] * 2
    }

    # Draw counting line and total count if visuals enabled
    if show_visuals:
        cv2.line(
            frame,
            (int(w / 2), int(h * 1 / 6)),
            (int(w / 2), int(h * 5 / 6)),
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Total Count: {total_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

    # Batch insert to database
    if len(batch_detections) >= CONFIG["BATCH_SIZE"] or frame_idx % 100 == 0:
        if batch_detections:
            db_insert(batch_detections)
            batch_detections.clear()

    # Encode JPEG and send via WebSocket
    _, buffer = cv2.imencode(".jpeg", frame)
    import base64

    frame_b64 = base64.b64encode(buffer).decode("utf-8")
    await websocket.send_text(frame_b64)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)