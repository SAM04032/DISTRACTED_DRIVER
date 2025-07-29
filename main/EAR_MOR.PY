import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Eye & mouth landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
OUTER_LIPS = [78, 308, 13, 14, 17, 82, 87, 312, 317, 402, 318, 324]

# Thresholds
EAR_THRESHOLD = 0.25
MOR_THRESHOLD = 0.6

def euclidean(p1, p2):
    return dist.euclidean(p1, p2)

def eye_aspect_ratio(landmarks, eye_points):
    A = euclidean(landmarks[eye_points[1]], landmarks[eye_points[5]])
    B = euclidean(landmarks[eye_points[2]], landmarks[eye_points[4]])
    C = euclidean(landmarks[eye_points[0]], landmarks[eye_points[3]])
    return (A + B) / (2.0 * C)

def mouth_opening_ratio(landmarks):
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]
    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)
    return vertical / horizontal

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_points = results.multi_face_landmarks[0].landmark
        landmarks = [(int(p.x * w), int(p.y * h)) for p in mesh_points]

        # EAR and MOR
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        mor = mouth_opening_ratio(landmarks)

        # Draw eyes and lips
        for idx in LEFT_EYE + RIGHT_EYE + OUTER_LIPS:
            cv2.circle(frame, landmarks[idx], 1, (0, 255, 0), -1)

        # Instant detection (no delay)
        if avg_ear < EAR_THRESHOLD:
            cv2.putText(frame, "Eyes Closed!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if mor > MOR_THRESHOLD:
            cv2.putText(frame, "Yawning!", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show EAR & MOR values
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (300, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"MOR: {mor:.2f}", (300, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Yawn & Eye Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
