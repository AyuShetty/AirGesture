import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize MediaPipe Drawing utilities to visualize hand landmarks
mp_drawing = mp.solutions.drawing_utils

# Set up camera
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, replace with your webcam device ID

# Define screen dimensions (adjust based on the projector resolution)
screen_width, screen_height = 1920, 1080  # Example: Full HD resolution

# Coordinates for the 4 corners of the iPad (can be adjusted based on your setup)
ipad_corners = []  # Initially empty, will be filled by user clicks
projected_corners = [(0, 0), (screen_width, 0), (screen_width, screen_height), (0, screen_height)]  # Projection corners

# Initialize perspective transformation variables
pts2 = np.array(projected_corners, dtype="float32")  # Destination points on the projected screen
pts1 = []  # Initialize as an empty list instead of None

def map_to_screen(x, y, frame_width, frame_height):
    """
    Map coordinates from camera frame to screen projection coordinates.
    """
    screen_x = int(x * screen_width / frame_width)
    screen_y = int(y * screen_height / frame_height)
    return screen_x, screen_y

def draw_corners(frame, corners, w, h):
    """
    Draw the four corner markers on the iPad screen in the camera feed.
    """
    for corner in corners:
        x, y = corner
        # Map the normalized corner coordinates to screen coordinates
        screen_x, screen_y = map_to_screen(x, y, w, h)
        # Draw a red marker at each corner
        cv2.circle(frame, (screen_x, screen_y), 10, (0, 0, 255), cv2.FILLED)  # Red corner markers

def apply_perspective_correction(frame):
    """
    Apply perspective correction to the frame using homography transformation.
    """
    global pts1
    if len(pts1) < 4:
        return frame  # If corners are not selected, return the original frame
    
    # Calculate the homography matrix
    h_matrix, _ = cv2.findHomography(np.array(pts1, dtype="float32"), pts2)
    
    # Apply the perspective warp
    corrected_frame = cv2.warpPerspective(frame, h_matrix, (screen_width, screen_height))
    
    return corrected_frame

def select_corners(event, x, y, flags, param):
    """
    Callback function to select the 4 corners of the iPad screen.
    """
    global pts1
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts1) < 4:
            pts1.append([x, y])
            cv2.circle(frame, (x, y), 10, (255, 0, 0), cv2.FILLED)
            cv2.imshow("Select Corners", frame)
            if len(pts1) == 4:
                print("Four corners selected!")
                cv2.destroyAllWindows()

# Set up a mouse callback for selecting corners
cv2.namedWindow("Select Corners")
cv2.setMouseCallback("Select Corners", select_corners)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Wait for the user to select the 4 corners of the iPad
    if len(pts1) < 4:
        # Show the live camera feed for corner selection
        cv2.imshow("Select Corners", frame)
        cv2.waitKey(1)
        continue
    
    # Once 4 corners are selected, apply perspective correction
    corrected_frame = apply_perspective_correction(frame)
    
    # Draw the 4 corners on the corrected frame
    draw_corners(corrected_frame, ipad_corners, w, h)
    
    # Convert image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Draw the hand landmarks and finger pointer on the corrected frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the index finger tip (Landmark 8)
            x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w
            y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h
            
            # Map the coordinates to the screen size (the projected screen)
            screen_x, screen_y = map_to_screen(x, y, w, h)
            
            # Draw the pointer (circle) at the finger tip position
            cv2.circle(corrected_frame, (screen_x, screen_y), 10, (255, 0, 0), cv2.FILLED)  # Blue pointer
            
            # Optionally draw the hand landmarks for visualization
            mp_drawing.draw_landmarks(corrected_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with perspective correction and hand tracking
    cv2.imshow('Finger Tracker', corrected_frame)
    
    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
