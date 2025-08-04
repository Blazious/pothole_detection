# detect.py
import os
import cv2
import torch
from ultralytics import YOLO
import tempfile
from PIL import Image
import numpy as np

# Load model once
model = YOLO('model/best.pt')

# Load beep
beep_path = os.path.join('media', 'beep.wav')

def detect_image(img: Image.Image):
    frame = np.array(img)

    results = model.predict(source=frame, conf=0.4)
    detections = results[0].boxes

    detected = False

    for box in detections:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"Pothole: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        detected = True

    # Save to temp file
    result_path = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(result_path, frame)

    return result_path, beep_path if detected else None

# detect.py (for videos)
def detect_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = tempfile.mktemp(suffix=".mp4")
    out = None
    detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.4)
        boxes = results[0].boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"Pothole: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detected = True

        if out is None:
            height, width, _ = frame.shape
            out = cv2.VideoWriter(out_path, fourcc, 20.0, (width, height))

        out.write(frame)

    cap.release()
    if out:
        out.release()

    return out_path, beep_path if detected else None

