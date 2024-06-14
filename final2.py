import cv2
import torch
import numpy as np
import mediapipe as mp
from datetime import datetime
from collections import Counter
from sklearn.cluster import KMeans
import webcolors
import csv

# Load the custom-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# MediaPipe hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Drawing utilities for MediaPipe
mp_drawing = mp.solutions.drawing_utils

# Flask application setup

# CSV file path
csv_file_path = "touch_events.csv"

# Update this list to include the labels of your custom-trained objects
object_labels = ['cell phone']  # Replace with your custom object labels

def get_dominant_color(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    
    pixels = roi.reshape((-1, 3))
    pixels = pixels[(pixels > 50).all(axis=1)]
    if pixels.size == 0:
        return (0, 0, 0)
    
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    
    return tuple(dominant_color)

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_NAMES_TO_HEX.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def convert_rgb_to_names(rgb_color):
    try:
        color_name = webcolors.rgb_to_name(rgb_color)
    except ValueError:
        color_name = closest_color(rgb_color)
    return color_name

def detect_and_log_touch_events(frame):
    results = model(frame)
    detections = results.xyxy[0].numpy()

    objects = []

    for detection in detections:
        x1, y1, x2, y2, confidence, cls = detection
        label = model.names[int(cls)]
        bbox = (x1, y1, x2, y2)

        if label in object_labels:
            objects.append((bbox, label))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)

    hand_bboxes = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

            hand_bboxes.append((x_min, y_min, x_max, y_max))

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(frame, "hand", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for hand_bbox in hand_bboxes:
        hx1, hy1, hx2, hy2 = hand_bbox
        for obj in objects:
            (ox1, oy1, ox2, oy2), obj_label = obj
            if (hx1 < ox2 and hx2 > ox1 and hy1 < oy2 and hy2 > oy1):
                touch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                obj_color = get_dominant_color(frame, (ox1, oy1, ox2, oy2))
                obj_hex = webcolors.rgb_to_hex(obj_color)
                # obj_name = convert_rgb_to_names(obj_color)
                print(f"Touch detected at {touch_time} on {obj_label} with color {obj_hex}")

                with open(csv_file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([touch_time, obj_label, obj_hex, obj_hex])

                cv2.rectangle(frame, (int(ox1), int(oy1)), (int(ox2), int(oy2)), 2)
                cv2.putText(frame, f"{obj_label}", (int(ox1), int(oy1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

    return frame



# Video Capture from file or webcam
cap = cv2.VideoCapture(0)  # Replace with 0 for webcam or path to video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = detect_and_log_touch_events(frame)
    cv2.imshow('Object and Hand Detection', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if __name__ == '__main__':
    # Initialize CSV file with headers if it doesn't exist
    try:
        with open(csv_file_path, 'x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Label", "Color Name", "Color Hex"])
    except FileExistsError:
        pass
    
    # Start the Flask server
    # app.run(debug=True)