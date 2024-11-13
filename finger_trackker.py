import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Set up camera
cap = cv2.VideoCapture(0)

# Initialize variables for corner calibration
calibrated = False
ipad_corners = []

def click_event(event, x, y, flags, param):
    global ipad_corners, calibrated
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(ipad_corners) < 4:
            ipad_corners.append((x, y))
            print(f"Corner {len(ipad_corners)} set at: {x}, {y}")
        if len(ipad_corners) == 4:
            calibrated = True

# Start the main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame for mirror effect and get dimensions
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Step 1: Calibration - Set up corners of iPad screen
    if not calibrated:
        # Display instruction text
        cv2.putText(frame, "Click the four corners of the iPad on screen", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show the points if any have been selected
        for corner in ipad_corners:
            cv2.circle(frame, corner, 5, (0, 0, 255), -1)

        # Set up mouse callback for selecting points
        cv2.imshow('Calibration', frame)
        cv2.setMouseCallback('Calibration', click_event)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Step 2: Define the target iPad screen dimensions (e.g., Full HD resolution for projection)
    ipad_width, ipad_height = 1920, 1080  # Define as per your projection display
    pts1 = np.float32(ipad_corners)
    pts2 = np.float32([[0, 0], [ipad_width, 0], [ipad_width, ipad_height], [0, ipad_height]])

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Check for hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get fingertip position of index finger (Landmark 8)
            x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w
            y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h

            # Transform finger coordinates to the iPad screen space
            input_coords = np.array([[x, y]], dtype='float32')
            input_coords = np.array([input_coords])
            ipad_coords = cv2.perspectiveTransform(input_coords, matrix)
            screen_x, screen_y = int(ipad_coords[0][0][0]), int(ipad_coords[0][0][1])

            # Draw the pointer circle on the frame at the fingertip position
            cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), cv2.FILLED)
            
            # Optionally draw the landmarks for visualization
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Finger Tracker', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
