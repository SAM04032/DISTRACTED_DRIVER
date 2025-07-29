import cv2
import pygame
import imutils
import numpy as np
import time
import dlib
from scipy.spatial import distance as dist  # Import the distance module

# Initialize pygame mixer for alert sound
pygame.mixer.init()
try:
    pygame.mixer.music.load("./music.wav")
    print("Sound file loaded successfully")
except:
    print("Could not load 'music.wav'. Make sure it exists in the same directory.")
    exit()

# YAWN DETECTION PARAMETERS
YAWN_THRESHOLD = 0.9  # Adjust this value
YAWN_CONSEC_FRAMES = 16 # Adjust this value
yawn_counter = 0

# Initialize dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat") # Ensure this path is correct!

def mouth_opening_ratio(landmarks):
    # ... (Your mouth_opening_ratio function code here) ...
    top_lip_mid = landmarks[50]
    bottom_lip_mid = landmarks[58]
    left_corner = landmarks[48]
    right_corner = landmarks[54]
    top_lip_upper = landmarks[51]
    bottom_lip_lower = landmarks[57]
    top_lip_inner = landmarks[52]
    bottom_lip_inner = landmarks[56]

    vertical1 = dist.euclidean(top_lip_mid, bottom_lip_mid)
    vertical2 = dist.euclidean(top_lip_upper, bottom_lip_lower)
    vertical3 = dist.euclidean(top_lip_inner, bottom_lip_inner)

    horizontal = dist.euclidean(left_corner, right_corner)

    mor = (vertical1 + vertical2 + vertical3) / (3.0 * horizontal) if horizontal > 0 else 0
    return mor

# Use webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Webcam not accessible")
    exit()

print("Yawn Detection Started (Landmark Based) - Press ESC to exit")
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = imutils.resize(frame, width=450)
    gray_cv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    # --- TESTING DLIB LOAD ---
    cv2.imwrite("temp_gray.png", gray_cv)
    dlib_gray_rgb = dlib.load_rgb_image("temp_gray.png")
    faces = detector(dlib_gray_rgb, 0)
    # --- END TESTING DLIB LOAD ---

    if not faces:  # Check if any faces were detected
        cv2.putText(frame, "No Face Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        yawn_counter = 0  # Reset yawn counter if no face is detected
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    else:
        for face in faces:
            landmarks = predictor(dlib_gray_rgb, face) # Use dlib_gray_rgb here too
            landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])


            mor_value = mouth_opening_ratio(landmarks)

            cv2.putText(frame, f"MOR: {mor_value:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if mor_value > YAWN_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= YAWN_CONSEC_FRAMES:
                    cv2.putText(frame, "YAWN DETECTED!", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play(-1)
                cv2.putText(frame, f"Yawning Frames: {yawn_counter}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                yawn_counter = 0
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()

            # You can also draw landmarks on the face for visualization (optional)
            # for (x, y) in landmarks:
            #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Yawn Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()