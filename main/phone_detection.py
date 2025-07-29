import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Load YOLOv8 model (use yolov8m or yolov8x for better accuracy)
model = YOLO('yolov8m.pt')

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)

# Object label for phone/cellphone
PHONE_LABELS = ['cell phone', 'mobile phone', 'telephone']

# Distance threshold
OBJECT_EAR_THRESHOLD = 120
HAND_EAR_THRESHOLD = 80


def is_near(point1, point2, threshold):
    x1, y1 = point1
    x2, y2 = point2
    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance < threshold


def check_hand_near_ear(hand_landmarks, ear_coords, w, h, frame, threshold):
    # Track wrist and fingertip proximity to ear
    tips = [0, 4, 8, 12, 16, 20]  # wrist and finger tips
    ex, ey = ear_coords
    for tip_id in tips:
        point = hand_landmarks.landmark[tip_id]
        px = int(point.x * w)
        py = int(point.y * h)
        cv2.circle(frame, (px, py), 4, (255, 255, 0), -1)

        if is_near((px, py), (ex, ey), threshold):
            return True
    return False


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(rgb)

    phone_detected = False
    phone_near_head = False
    hand_near_head = False

    # Estimate ear/cheek region using face landmarks
    ear_x, ear_y = -1, -1
    if results.face_landmarks:
        left_ear = results.face_landmarks.landmark[234]  # near left cheek
        right_ear = results.face_landmarks.landmark[454]  # near right cheek
        ear_x = int((left_ear.x + right_ear.x) / 2 * w)
        ear_y = int((left_ear.y + right_ear.y) / 2 * h)
        cv2.circle(frame, (ear_x, ear_y), 6, (0, 255, 0), -1)

    # YOLOv8 Object Detection
    yolo_results = model(frame, verbose=False)

    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in PHONE_LABELS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                phone_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # Check if phone near ear
                if ear_x != -1 and is_near((cx, cy), (ear_x, ear_y), OBJECT_EAR_THRESHOLD):
                    phone_near_head = True

    # Hand-to-ear detection using hand landmarks
    if results.left_hand_landmarks and ear_x != -1:
        if check_hand_near_ear(results.left_hand_landmarks, (ear_x, ear_y), w, h, frame, HAND_EAR_THRESHOLD):
            hand_near_head = True

    if results.right_hand_landmarks and ear_x != -1:
        if check_hand_near_ear(results.right_hand_landmarks, (ear_x, ear_y), w, h, frame, HAND_EAR_THRESHOLD):
            hand_near_head = True

    # Decision display
    if phone_detected and phone_near_head:
        cv2.putText(frame, "Phone Use Detected (Object)", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    if hand_near_head:
        cv2.putText(frame, "Phone Use Detected (Hand Near Ear)", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # Display result
    cv2.imshow("Phone/Calling Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
