import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
corner_count = 0
pts1 = []  # List to store corner points
matrix = None
hover_start_time = None
hover_duration = 5  # Set the time for hovering over a corner (in seconds)
fingertip_history = deque(maxlen=5)  # Track fingertip positions
pointer_history = deque(maxlen=10)  # History for smoothing pointer movement
last_time = time.time()

def draw_markers(frame, points):
    """Draw markers for the registered corners."""
    for i, pt in enumerate(points):
        pt = tuple(map(int, pt))
        cv2.circle(frame, pt, 10, (255, 0, 0), -1)
        cv2.putText(frame, f"Corner {i + 1}", (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def map_to_screen(x, y, matrix):
    """Map hand coordinates to screen coordinates using perspective transform."""
    point = np.array([[x, y]], dtype='float32')
    point = np.array([point])
    transformed_point = cv2.perspectiveTransform(point, matrix)
    return int(transformed_point[0][0][0]), int(transformed_point[0][0][1])

def calculate_scaling_factor(current_pos, fingertip_history, dt):
    """Calculate the movement scaling factor based on speed."""
    if len(fingertip_history) < 2:
        return 1

    prev_pos = fingertip_history[-2]
    distance = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
    speed = distance / dt if dt > 0 else 0

    # Adjust scaling factor based on speed
    base_scale = 1
    scale = min(max(base_scale * (speed / 15), 0.8), 2)  # Clamp between 0.8 and 2
    return scale

def smooth_pointer(pointer_history):
    """Smooth pointer position using averaging."""
    if not pointer_history:
        return None
    avg_x = int(np.mean([pos[0] for pos in pointer_history]))
    avg_y = int(np.mean([pos[1] for pos in pointer_history]))
    return avg_x, avg_y

def auto_detect_corners(frame):
    """Automatically detect the four corners of the screen."""
    h, w, _ = frame.shape
    return [
        (50, 50),         # Top Left
        (w - 50, 50),     # Top Right
        (50, h - 50),     # Bottom Left
        (w - 50, h - 50)  # Bottom Right
    ]

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

print("Do you want to recognize corners automatically? (y/n)")
manual_mode = input().strip().lower() != 'y'

if not manual_mode:
    print("Automatic mode selected! Corners will be detected automatically.")
else:
    print("Manual mode selected! Hover over each corner for 5 seconds to capture it.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if not manual_mode and corner_count == 0:
        pts1 = auto_detect_corners(frame)
        print("Automatic corner registration complete!")
        for i, corner in enumerate(["Top Left", "Top Right", "Bottom Left", "Bottom Right"]):
            print(f"{corner} registered at: {pts1[i]}")
        corner_count = 4
        matrix = cv2.getPerspectiveTransform(
            np.array(pts1, dtype="float32"),
            np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype="float32")
        )
        print("Perspective transformation matrix calculated!")

    if manual_mode and corner_count < 4:
        message = ["Top Left", "Top Right", "Bottom Left", "Bottom Right"][corner_count]
        cv2.putText(frame, f"Hover over the {message} corner to capture.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        fingertip = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingertip = (
                    int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                    int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
                )

        if fingertip:
            if hover_start_time is None:
                hover_start_time = time.time()
            elif time.time() - hover_start_time >= hover_duration:
                pts1.append(fingertip)
                print(f"{message} registered at: {pts1[-1]}")
                corner_count += 1
                hover_start_time = None
        else:
            hover_start_time = None

        if pts1:
            draw_markers(frame, pts1)

    if manual_mode and corner_count == 4 and matrix is None:
        print("Corner registration complete!")
        pts1 = np.array(pts1, dtype="float32")
        pts2 = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        print("Perspective transformation matrix calculated!")

    if matrix is not None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        fingertip = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingertip = (
                    int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                    int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
                )

        if fingertip:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            fingertip_history.append(fingertip)
            scale = calculate_scaling_factor(fingertip, fingertip_history, dt)

            screen_x, screen_y = map_to_screen(fingertip[0] * scale, fingertip[1] * scale, matrix)
            screen_x = min(max(screen_x, 0), w - 1)
            screen_y = min(max(screen_y, 0), h - 1)

            pointer_history.append((screen_x, screen_y))
            smoothed_pointer = smooth_pointer(pointer_history)
            if smoothed_pointer:
                cv2.circle(frame, smoothed_pointer, 10, (0, 0, 255), -1)

    cv2.imshow("Hand Tracking and Pointer", frame)
