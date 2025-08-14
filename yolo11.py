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

#========USING GPU IF AVAILABLE=======#
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#========SET UP ENV FOR RTSP STREAM (USING TCP)==========#
import os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'


#========CONFIG VARIABLES==========#
RTSP_URL = "rtsp://10.6.18.5:46458/mystream" # RTSP stream URL
CONF_THRESHOLD = 0.5 # Confidence threshold for detections
MOVE_THRESHOLD = 5 # Threshold for movement detection (move at least 5 pixels)
MIN_LIFETIME = 3 # Minimum lifetime of a track (in frames)

total_count = 0 # Total count of detected objects


#========MODEL LOADING========#
best_model = YOLO('D:/Test/best.pt').to(device)


#========STREAM READING FOR FPS========#
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
fps = cap.get(cv2.CAP_PROP_FPS)

#========TRACKING (USING BYTETRACK)========#
tracker = ByteTrack(frame_rate=int(fps), lost_track_buffer=100)
track_memory = {}       # Store information for each track_id
frame_idx = 0

#=======FRAME READING USING THREAD========#
# Queue to store frames
frame_queue = queue.Queue(maxsize=128)

# Thread to read frames from the RTSP stream
def frame_reader(rtsp_url, q):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Thread error: Unable to open stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Thread error: Failed to read frame or stream ended.")
            break                                                           
        # If frame is read successfully and queue is not full, add it to the queue
        if not q.full():
            q.put(frame)
    cap.release()

# Start the frame reader thread 
reader_thread = threading.Thread(target=frame_reader, args=(RTSP_URL, frame_queue))
reader_thread.daemon = True # Thread sẽ tự tắt khi chương trình chính kết thúc
reader_thread.start()

#========STREAM PROCESSING========#
while True:
    # Read frame from queue
    if not frame_queue.empty():
        frame = frame_queue.get()

        frame_idx += 1

        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Resize frame if needed
        h, w = frame.shape[:2]
        h = h * 60 // 100
        w = w * 60 // 100
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        #=======DETECTION=======#
        results = best_model(frame, verbose=False, conf=0.7)[0]
        detections = Detections.from_ultralytics(results)

        #=======TRACKING=======#
        tracked = tracker.update_with_detections(detections)

        # ID hiện tại trong frame
        current_ids = set()

        for i in range(len(tracked)):
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            track_id = int(tracked.tracker_id[i])
            class_id = int(tracked.class_id[i])
            confidence = float(tracked.confidence[i])

            current_ids.add(track_id)

            # Chỉ xử lý class stamp (giả sử class_id = 1)
            if class_id != 1 or confidence < CONF_THRESHOLD:
                # still update last_seen for this track
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

                moved_enough = abs(prev_cx - cx) >= MOVE_THRESHOLD or abs(prev_cy - cy) >= MOVE_THRESHOLD
                crossed_vertical = (prev_cx < int(w/2 + 5) and cx >= int(w/2 + 5) or prev_cx >= int(w/2 + 5) and cx < int(w/2 + 5))
                in_vertical_y_range = (cy >= int(h * 1/6) and cy <= int(h * 5/6))

                # Đếm khi đủ tuổi thọ, di chuyển, và qua vạch
                if lifetime >= MIN_LIFETIME and not track_memory[track_id]["counted"] and moved_enough and crossed_vertical and in_vertical_y_range and not track_memory[track_id]["passed_vertical"]:
                    total_count += 1
                    track_memory[track_id]["counted"] = True
                    track_memory[track_id]["passed_vertical"] = True

                    #=======SAVE DETECTION=======#
                    frame_detections = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "track_id": track_id,
                        "label": results.names[class_id],
                        "confidence": round(confidence, 3),
                        "bbox": [x1, y1, x2, y2]
                    }

                    # Chỉ insert khi có detection mới
                    if frame_detections:
                        db_insert([frame_detections])

                # Cập nhật toạ độ
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

        # Xoá track_id không còn trong frame
        track_memory = {tid: m for tid, m in track_memory.items() if frame_idx - m.get('last_seen', frame_idx) <= 100}

        # Vẽ vạch kiểm tra
        cv2.line(frame, (int(w/2), int(h * 1/6)), (int(w/2), int(h * 5/6)), (0, 255, 255), 3)
        # Hiển thị tổng count
        cv2.putText(frame, f"Total Count: {total_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # Show frame
        cv2.imshow("RTSP Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()