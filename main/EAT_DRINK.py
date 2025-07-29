import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Load YOLOv8 model (consider using yolov8x.pt for higher accuracy)
model = YOLO('yolov8m.pt')

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)

# Labels relevant to eating/drinking
EAT_DRINK_LABELS = ['bottle', 'cup', 'wine glass', 'fork', 'spoon', 'bowl',
                    'sandwich', 'banana', 'apple', 'orange', 'hot dog', 'pizza', 'cake']

# Constants
OBJECT_MOUTH_THRESHOLD = 100
HAND_MOUTH_THRESHOLD = 50
YOLO_CONFIDENCE = 0.4  # Minimum confidence for valid detection


def is_near_mouth(mouth_coords, obj_box, threshold=100):
    mx, my = mouth_coords
    x1, y1, x2, y2 = obj_box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    distance = np.sqrt((mx - cx) ** 2 + (my - cy) ** 2)

    vertical_check = cy > my - 30  # object should be at or below the mouth level
    return distance < threshold and vertical_check


def check_hand_to_mouth(landmarks, mouth_coords, w, h, frame, threshold=HAND_MOUTH_THRESHOLD):
    tips = [4, 8, 12, 16, 20]  # Finger tips: Thumb to Pinky
    mx, my = mouth_coords
    for tip_id in tips:
        tip = landmarks.landmark[tip_id]
        tip_x = int(tip.x * w)
        tip_y = int(tip.y * h)
        cv2.circle(frame, (tip_x, tip_y), 4, (255, 255, 0), -1)

        dist = np.sqrt((tip_x - mx) ** 2 + (tip_y - my) ** 2)
        if dist < threshold:
            return True
    return False


def mouth_opening_ratio(landmarks, w, h):
    if 13 >= len(landmarks.landmark) or 14 >= len(landmarks.landmark):
        return 0
    top = landmarks.landmark[13]
    bottom = landmarks.landmark[14]
    return np.sqrt((top.x - bottom.x) ** 2 + (top.y - bottom.y) ** 2)


# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get face and hand landmarks
    results = holistic.process(rgb)

    # Initialize variables
    mouth_x, mouth_y = -1, -1
    eating_detected = False
    hand_to_mouth = False
    mouth_opening = False

    # Extract mouth center
    if results.face_landmarks:
        mouth_top = results.face_landmarks.landmark[13]
        mouth_bottom = results.face_landmarks.landmark[14]
        mouth_x = int((mouth_top.x + mouth_bottom.x) / 2 * w)
        mouth_y = int((mouth_top.y + mouth_bottom.y) / 2 * h)
        cv2.circle(frame, (mouth_x, mouth_y), 5, (0, 255, 0), -1)

        # Calculate mouth opening ratio
        mor = mouth_opening_ratio(results.face_landmarks, w, h)
        if mor > 0.03:  # Adjust threshold as needed
            mouth_opening = True

    # YOLO Object Detection
    yolo_results = model(frame, verbose=False)

    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < YOLO_CONFIDENCE:
                continue

            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in EAT_DRINK_LABELS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Check proximity to mouth
                if mouth_x != -1 and is_near_mouth((mouth_x, mouth_y), (x1, y1, x2, y2), OBJECT_MOUTH_THRESHOLD):
                    if mouth_opening or hand_to_mouth:
                        eating_detected = True

    # Hand-to-mouth from either hand
    if results.left_hand_landmarks and mouth_x != -1:
        if check_hand_to_mouth(results.left_hand_landmarks, (mouth_x, mouth_y), w, h, frame):
            hand_to_mouth = True

    if results.right_hand_landmarks and mouth_x != -1:
        if check_hand_to_mouth(results.right_hand_landmarks, (mouth_x, mouth_y), w, h, frame):
            hand_to_mouth = True

    # Display detection results
    if eating_detected:
        cv2.putText(frame, "Eating/Drinking Detected", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    elif hand_to_mouth:
        cv2.putText(frame, "Hand-to-Mouth Detected", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    elif mouth_opening:
        cv2.putText(frame, "Mouth Open (Possibly Eating)", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Eating/Drinking Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
