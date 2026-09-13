import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Function to calculate EAR (Eye Aspect Ratio)
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

# Constants
EYE_AR_THRESH = 0.25  # EAR threshold for a blink
EYE_AR_CONSEC_FRAMES = 3  # Frames to consider for a blink

# Initialize variables
blink_counter = 0
total_blinks = 0

# Load dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Landmark indices for eyes
(l_start, l_end) = (42, 48)  # Right eye
(r_start, r_end) = (36, 42)  # Left eye

# Start video stream
cap = cv2.VideoCapture(0)

print("[INFO] Starting camera feed. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Unable to access the camera.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        # Extract eye coordinates
        left_eye = landmarks[r_start:r_end]
        right_eye = landmarks[l_start:l_end]

        # Compute EAR for both eyes
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Draw the eyes for visualization
        cv2.polylines(frame, [np.array(left_eye, dtype=np.int32)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(right_eye, dtype=np.int32)], True, (0, 255, 0), 1)

        # Check if EAR is below threshold
        if ear < EYE_AR_THRESH:
            blink_counter += 0.25
        else:
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                total_blinks += 0.25
                print(f"[INFO] Blink detected! Total blinks: {total_blinks}")
            blink_counter = 0

        # Display EAR value on frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display total blinks
    cv2.putText(frame, f"Blinks: {total_blinks}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show video frame
    cv2.imshow("Eye Blink Detection", frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
