import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize MediaPipe Drawing utilities to visualize hand landmarks
mp_drawing = mp.solutions.drawing_utils

# Set up camera
cap = cv2.VideoCapture(0)

# Define screen dimensions (adjust based on the projector resolution)
screen_width, screen_height = 1920, 1080  # Example: Full HD resolution

def map_to_screen(x, y, frame_width, frame_height):
    """
    Map coordinates from camera frame to screen projection coordinates.
    """
    screen_x = int(x * screen_width / frame_width)
    screen_y = int(y * screen_height / frame_height)
    return screen_x, screen_y

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the index finger tip (Landmark 8)
            x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w
            y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h
            
            # Map the coordinates to the screen size
            screen_x, screen_y = map_to_screen(x, y, w, h)
            
            # Draw the pointer (circle) at the finger tip position
            cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), cv2.FILLED)
            
            # Optionally draw the hand landmarks for visualization
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with the finger position
    cv2.imshow('Finger Tracker', frame)
    
    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
