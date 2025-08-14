import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import cv2
from ultralytics import YOLO  # Detection Model
from supervision import ByteTrack, Detections # Tracking method
from datetime import datetime
import time
import threading
import queue
from mySQL_db import db_insert # Save data to db
import logging
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
    "RTSP_URL": "rtsp://10.6.18.5:46458/mystream",
    "MODEL_PATH": "D:/Test/best.pt",
    "CONF_THRESHOLD": 0.5,
    "MOVE_THRESHOLD": 5,
    "MIN_LIFETIME": 3,
    "QUEUE_SIZE": 128,
    "LOST_TRACK_BUFFER": 50,
    "BATCH_SIZE": 10,  # For database insertions
    "FRAME_SKIP": 1,  # Process every Nth frame (1 = no skip)
    "RESIZE_PERCENT": 60,
    "CLASS_ID": 1  # Assuming stamp class_id = 1
}

total_count = 0 # Total count of detected objects

#========MODEL LOADING========#
best_model = YOLO(CONFIG["MODEL_PATH"]).to(device)
try:
    best_model.model.half()  # Enable FP16 inference if supported
except:
    logger.warning("Mixed precision not supported on this device.")

#========STREAM READING FOR FPS========#
cap = cv2.VideoCapture(CONFIG["RTSP_URL"], cv2.CAP_FFMPEG)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()  # Release this cap as we'll use threaded reader

#========TRACKING (USING BYTETRACK)========#
tracker = ByteTrack(frame_rate=int(fps), lost_track_buffer=CONFIG["LOST_TRACK_BUFFER"])
track_memory = {}       # Store information for each track_id
frame_idx = 0

#=======FRAME READING USING THREAD========#
# Queue to store frames
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
            h = h * CONFIG["RESIZE_PERCENT"] // 100
            w = w * CONFIG["RESIZE_PERCENT"] // 100
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
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
def process_frame(frame, model, tracker, track_memory, frame_idx, w, h, batch_detections):
    with torch.no_grad():
        results = model(frame, verbose=False, conf=0.7)[0]
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

        # Only process specific class
        if class_id != CONFIG["CLASS_ID"] or confidence < CONFIG["CONF_THRESHOLD"]:
            if track_id in track_memory:
                track_memory[track_id]['last_seen'] = frame_idx
            continue

        #=======TRACK MEMORY UPDATE=======#
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

            # Count when conditions met
            if lifetime >= CONFIG["MIN_LIFETIME"] and not track_memory[track_id]["counted"] and moved_enough and crossed_vertical and in_vertical_y_range and not track_memory[track_id]["passed_vertical"]:
                global total_count
                total_count += 1
                track_memory[track_id]["counted"] = True
                track_memory[track_id]["passed_vertical"] = True

                # Prepare for batch insert
                frame_detections = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "track_id": track_id,
                    "label": results.names[class_id],
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2]
                }
                batch_detections.append(frame_detections)

            # Update coordinates
            track_memory[track_id]["prev_cx"] = track_memory[track_id]["cx"]
            track_memory[track_id]["prev_cy"] = track_memory[track_id]["cy"]
            track_memory[track_id]["cx"] = cx
            track_memory[track_id]["cy"] = cy
            track_memory[track_id]["lifetime"] += 1
            track_memory[track_id]["last_seen"] = frame_idx

        #=======DRAW=======#
        label = f"{results.names[class_id]} {track_id}: {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return tracked, current_ids, batch_detections

#========STREAM PROCESSING========#
batch_detections = []
show_visuals = True
while True:
    if not frame_queue.empty():
        start_time = time.time()
        frame = frame_queue.get()
        h, w = frame.shape[:2]
        frame_idx += 1

        # Frame skipping for detection
        if frame_idx % CONFIG["FRAME_SKIP"] == 0:
            tracked, current_ids, batch_detections = process_frame(frame, best_model, tracker, track_memory, frame_idx, w, h, batch_detections)
        else:
            # On skipped frames, update tracker with empty detections
            detections = Detections.empty()
            tracked = tracker.update_with_detections(detections)
            current_ids = set()

        # Clean up old tracks
        track_memory = {tid: m for tid, m in track_memory.items() if frame_idx - m.get('last_seen', frame_idx) <= CONFIG["LOST_TRACK_BUFFER"] * 2}

        # Draw counting line and total count if visuals enabled
        if show_visuals:
            cv2.line(frame, (int(w/2), int(h * 1/6)), (int(w/2), int(h * 5/6)), (0, 255, 255), 3)
            cv2.putText(frame, f"Total Count: {total_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # Show frame
        cv2.imshow("RTSP Live Detection", frame)

        # Toggle visuals with 'v', quit with 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v'):
            show_visuals = not show_visuals

        # Batch insert to database
        if len(batch_detections) >= CONFIG["BATCH_SIZE"] or frame_idx % 100 == 0:
            if batch_detections:
                db_insert(batch_detections)
                batch_detections = []

cv2.destroyAllWindows()