import cv2
import torch
import numpy as np
import mediapipe as mp
from datetime import datetime
from collections import Counter
from sklearn.cluster import KMeans
from webcolors import rgb_to_name, hex_to_name, rgb_to_hex
import webcolors

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

# MediaPipe hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Drawing utilities for MediaPipe
mp_drawing = mp.solutions.drawing_utils

csv_file_path = "touch_events.csv"

# Labels of interest
object_labels = ['cell phone', 'shoe', 'clothes']  # Add other object labels as needed

def get_dominant_color(image, bbox):
    """
    Calculate the dominant color of an object within the bounding box using KMeans clustering.
    """
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    
    # Reshape the image to be a list of pixels
    pixels = roi.reshape((-1, 3))
    
    # Remove black or nearly black pixels to avoid dark color bias
    pixels = pixels[(pixels > 50).all(axis=1)]
    if pixels.size == 0:
        return (0, 0, 0)
    
    # Use KMeans to find the most common color
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    
    return tuple(dominant_color)

def convert_rgb_to_names(rgb_color):
    """
    Convert an RGB color to the nearest human-readable color name.
    """
    try:
        # Try to get the exact color name from the RGB tuple
        color_name = rgb_to_name(rgb_color)
    except ValueError:
        # Find the closest color name by minimizing the Euclidean distance
        min_colors = {}
        for hex_code, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r, g, b = webcolors.hex_to_rgb(hex_code)
            rd = (r - rgb_color[0]) ** 2
            gd = (g - rgb_color[1]) ** 2
            bd = (b - rgb_color[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        color_name = min_colors[min(min_colors.keys())]
    return color_name

def detect_and_log_touch_events(frame):
    results = model(frame)
    detections = results.xyxy[0].numpy()  # Extract detections

    objects = []

    for detection in detections:
        x1, y1, x2, y2, confidence, cls = detection
        label = model.names[int(cls)]
        bbox = (x1, y1, x2, y2)

        if label in object_labels:
            objects.append((bbox, label))

    # Process frame for hand detection using MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    lower_range = (0, 50, 50) # lower range of red color in HSV
    upper_range = (10, 255, 255) # upper range of red color in HSV
    mask = cv2.inRange(frame_rgb, lower_range, upper_range)

    hand_results = hands.process(frame_rgb)

    hand_bboxes = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Calculate bounding box for each detected hand
            h, w, _ = frame.shape
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

            hand_bboxes.append((x_min, y_min, x_max, y_max))

            # Draw hand landmarks and bounding box
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(frame, "hand", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Check for touch events
    for hand_bbox in hand_bboxes:
        hx1, hy1, hx2, hy2 = hand_bbox
        for obj in objects:
            (ox1, oy1, ox2, oy2), obj_label = obj
            if (hx1 < ox2 and hx2 > ox1 and hy1 < oy2 and hy2 > oy1):  # Simple bounding box intersection check
                # Hand is touching the object
                touch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                obj_color = get_dominant_color(frame, (ox1, oy1, ox2, oy2))
                obj_hex = rgb_to_hex(obj_color)
                obj_name = convert_rgb_to_names(obj_color)
                # obj_color = get_object_color(frame, (ox1, oy1, ox2, oy2))
                print(f"Touch detected at {touch_time} on {obj_label} with color {obj_name} and hex {obj_hex}")

                # Draw bounding boxes and labels for objects
                cv2.rectangle(frame, (int(ox1), int(oy1)), (int(ox2), int(oy2)), 2)
                cv2.putText(frame, f"{obj_label}", (int(ox1), int(oy1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

    return frame

# Video Capture from file or webcam
cap = cv2.VideoCapture(0)  # Replace with 0 for webcam or path to video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    output_frame = detect_and_log_touch_events(frame)

    # Display the output frame
    cv2.imshow('Object and Hand Detection', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()