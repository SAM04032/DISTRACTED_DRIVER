import cv2
import numpy as np
import mediapipe as mp
import time
from scipy.spatial import distance as dist
from ultralytics import YOLO
import threading
import queue

class DriverMonitoringSystem:
    def __init__(self):
        # Initialize YOLO model with smaller version for better speed
        self.model = YOLO('yolov8n.pt')  # Use nano version instead of medium for speed
        
        # MediaPipe solutions with reduced complexity
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False, 
            model_complexity=0,  # Use lowest complexity
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Constants for all detection modules
        # Phone detection
        self.PHONE_LABELS = ['cell phone', 'mobile phone', 'telephone']
        self.OBJECT_EAR_THRESHOLD = 120
        self.HAND_EAR_THRESHOLD = 80
        
        # Eating/drinking detection
        self.EAT_DRINK_LABELS = ['bottle', 'cup', 'wine glass', 'fork', 'spoon', 'bowl',
                        'sandwich', 'banana', 'apple', 'orange', 'hot dog', 'pizza', 'cake']
        self.OBJECT_MOUTH_THRESHOLD = 100
        self.HAND_MOUTH_THRESHOLD = 50
        self.YOLO_CONFIDENCE = 0.4
        
        # Eye/mouth landmarks
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.OUTER_LIPS = [78, 308, 13, 14, 17, 82, 87, 312, 317, 402, 318, 324]
        self.EAR_THRESHOLD = 0.25
        self.MOR_THRESHOLD = 0.6
        
        # Head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),         # Nose tip
            (0.0, -330.0, -65.0),    # Chin
            (-225.0, 170.0, -135.0), # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),# Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])
        self.landmark_ids = [1, 152, 263, 33, 287, 57]
        
        # Detection state variables with timing
        self.detections = {
            'phone_use': False,
            'eating_drinking': False,
            'eyes_closed': False,
            'yawning': False,
            'head_position': 'Forward',
            'distracted': False
        }
        
        # Timers for persistent detections
        self.eyes_closed_start = None
        self.yawning_start = None
        self.EYES_CLOSED_THRESHOLD = 1.5  # seconds
        self.YAWNING_THRESHOLD = 1.0     # seconds
        
        # For status display - creates persistent overlay
        self.status_overlay = None
        self.last_status_update_time = 0
        self.STATUS_UPDATE_INTERVAL = 0.5  # seconds between status updates
        
        # Performance optimization
        self.frame_skip = 2  # Process every nth frame with YOLO
        self.frame_count = 0
        self.last_yolo_results = None
        
        # For async YOLO processing
        self.yolo_queue = queue.Queue(maxsize=1)
        self.yolo_results_queue = queue.Queue(maxsize=1)
        self.yolo_thread_active = False
        
    def start_yolo_thread(self):
        """Start a separate thread for YOLO processing"""
        self.yolo_thread_active = True
        yolo_thread = threading.Thread(target=self.yolo_worker)
        yolo_thread.daemon = True
        yolo_thread.start()
        
    def yolo_worker(self):
        """Worker thread for running YOLO detections"""
        while self.yolo_thread_active:
            try:
                if not self.yolo_queue.empty():
                    frame = self.yolo_queue.get(block=False)
                    results = self.model(frame, verbose=False)
                    if not self.yolo_results_queue.full():
                        self.yolo_results_queue.put(results)
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in YOLO thread: {e}")
        
    def is_near(self, point1, point2, threshold):
        x1, y1 = point1
        x2, y2 = point2
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distance < threshold
    
    def euclidean(self, p1, p2):
        return dist.euclidean(p1, p2)
    
    def eye_aspect_ratio(self, landmarks, eye_points):
        A = self.euclidean(landmarks[eye_points[1]], landmarks[eye_points[5]])
        B = self.euclidean(landmarks[eye_points[2]], landmarks[eye_points[4]])
        C = self.euclidean(landmarks[eye_points[0]], landmarks[eye_points[3]])
        return (A + B) / (2.0 * C)
    
    def mouth_opening_ratio(self, landmarks):
        if len(landmarks) <= 14 or len(landmarks) <= 78 or len(landmarks) <= 308:
            return 0
        top = landmarks[13]
        bottom = landmarks[14]
        left = landmarks[78]
        right = landmarks[308]
        vertical = self.euclidean(top, bottom)
        horizontal = self.euclidean(left, right)
        if horizontal == 0:
            return 0
        return vertical / horizontal
    
    def check_hand_near_point(self, hand_landmarks, point_coords, w, h, frame, threshold):
        tips = [0, 4, 8, 12, 16, 20]  # wrist and finger tips
        px, py = point_coords
        for tip_id in tips:
            point = hand_landmarks.landmark[tip_id]
            tx = int(point.x * w)
            ty = int(point.y * h)
            
            # Skip drawing in every frame for performance
            if self.frame_count % 3 == 0:
                cv2.circle(frame, (tx, ty), 4, (255, 255, 0), -1)
            
            if self.is_near((tx, ty), (px, py), threshold):
                return True
        return False
    
    def is_near_mouth(self, mouth_coords, obj_box, threshold=100):
        mx, my = mouth_coords
        x1, y1, x2, y2 = obj_box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        distance = np.sqrt((mx - cx) ** 2 + (my - cy) ** 2)
        
        vertical_check = cy > my - 30  # object should be at or below the mouth level
        return distance < threshold and vertical_check
    
    def process_phone_detection(self, frame, results, ear_coords, yolo_results=None):
        phone_detected = False
        phone_near_head = False
        hand_near_head = False
        h, w = frame.shape[:2]
        
        # YOLOv8 Object Detection for phones
        if yolo_results is not None:
            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    
                    if label in self.PHONE_LABELS:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        
                        phone_detected = True
                        
                        # Only draw rectangles occasionally to save processing
                        if self.frame_count % 3 == 0:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            cv2.putText(frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        
                        # Check if phone near ear
                        if ear_coords[0] != -1 and self.is_near((cx, cy), ear_coords, self.OBJECT_EAR_THRESHOLD):
                            phone_near_head = True
        
        # Hand-to-ear detection only check every few frames
        if self.frame_count % 2 == 0:
            if results.left_hand_landmarks and ear_coords[0] != -1:
                if self.check_hand_near_point(results.left_hand_landmarks, ear_coords, w, h, frame, self.HAND_EAR_THRESHOLD):
                    hand_near_head = True
                    
            if results.right_hand_landmarks and ear_coords[0] != -1:
                if self.check_hand_near_point(results.right_hand_landmarks, ear_coords, w, h, frame, self.HAND_EAR_THRESHOLD):
                    hand_near_head = True
        
        return phone_detected and phone_near_head, hand_near_head
    
    def process_eating_drinking(self, frame, results, mouth_coords, yolo_results=None):
        eating_detected = False
        hand_to_mouth = False
        mouth_opening = False
        h, w = frame.shape[:2]
        
        # Calculate mouth opening ratio if face landmarks are detected
        if results.face_landmarks:
            try:
                landmarks = [(int(p.x * w), int(p.y * h)) for p in results.face_landmarks.landmark]
                mor = self.mouth_opening_ratio(landmarks)
                if mor > 0.03:  # Adjust threshold as needed
                    mouth_opening = True
            except IndexError:
                # Skip if landmarks aren't properly detected
                pass
        
        # YOLO Object Detection for food/drink items - only if results available
        if yolo_results is not None:
            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < self.YOLO_CONFIDENCE:
                        continue
                        
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    
                    if label in self.EAT_DRINK_LABELS:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Only draw occasionally
                        if self.frame_count % 3 == 0:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(frame, f"{label}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        # Check proximity to mouth
                        if mouth_coords[0] != -1 and self.is_near_mouth(mouth_coords, (x1, y1, x2, y2), self.OBJECT_MOUTH_THRESHOLD):
                            if mouth_opening or hand_to_mouth:
                                eating_detected = True
        
        # Check hand-to-mouth only every other frame
        if self.frame_count % 2 == 0:
            # Hand-to-mouth detection
            if results.left_hand_landmarks and mouth_coords[0] != -1:
                if self.check_hand_near_point(results.left_hand_landmarks, mouth_coords, w, h, frame, self.HAND_MOUTH_THRESHOLD):
                    hand_to_mouth = True
                    
            if results.right_hand_landmarks and mouth_coords[0] != -1:
                if self.check_hand_near_point(results.right_hand_landmarks, mouth_coords, w, h, frame, self.HAND_MOUTH_THRESHOLD):
                    hand_to_mouth = True
        
        return eating_detected, hand_to_mouth, mouth_opening
    
    def process_ear_mor(self, frame, results):
        h, w = frame.shape[:2]
        eyes_closed = False
        yawning = False
        
        # Skip processing occasionally for better performance
        if self.frame_count % 2 != 0:
            return self.detections['eyes_closed'], self.detections['yawning']
        
        if results.face_landmarks:
            try:
                landmarks = [(int(p.x * w), int(p.y * h)) for p in results.face_landmarks.landmark]
                
                # EAR and MOR calculation
                left_ear = self.eye_aspect_ratio(landmarks, self.LEFT_EYE)
                right_ear = self.eye_aspect_ratio(landmarks, self.RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0
                mor = self.mouth_opening_ratio(landmarks)
                
                # Reduce visualization for better performance
                if self.frame_count % 4 == 0:
                    # Draw just a subset of points
                    for idx in self.LEFT_EYE[:2] + self.RIGHT_EYE[:2] + self.OUTER_LIPS[:2]:
                        cv2.circle(frame, landmarks[idx], 1, (0, 255, 0), -1)
                
                # Detection logic
                if avg_ear < self.EAR_THRESHOLD:
                    # Eyes closed detection - start timer if not already started
                    if not self.eyes_closed_start:
                        self.eyes_closed_start = time.time()
                    
                    # Only trigger if eyes closed for threshold duration
                    if time.time() - self.eyes_closed_start >= self.EYES_CLOSED_THRESHOLD:
                        eyes_closed = True
                else:
                    # Reset timer if eyes open
                    self.eyes_closed_start = None
                
                if mor > self.MOR_THRESHOLD:
                    # Yawning detection - start timer if not already started
                    if not self.yawning_start:
                        self.yawning_start = time.time()
                    
                    # Only trigger if yawning for threshold duration
                    if time.time() - self.yawning_start >= self.YAWNING_THRESHOLD:
                        yawning = True
                else:
                    # Reset timer if mouth closed
                    self.yawning_start = None
                
                # Show EAR & MOR values less frequently
                if self.frame_count % 5 == 0:
                    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (300, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"MOR: {mor:.2f}", (300, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            except (IndexError, ZeroDivisionError):
                # Skip if landmarks calculation fails 
                pass
                
        return eyes_closed, yawning
    
    def process_head_position(self, frame, results):
        h, w = frame.shape[:2]
        head_position = "Forward"
        
        # Skip occasionally for better performance
        if self.frame_count % 3 != 0:
            return self.detections['head_position']
        
        if results.face_landmarks:
            try:
                # Get key face landmarks for pose estimation
                image_points = []
                for idx in self.landmark_ids:
                    pt = results.face_landmarks.landmark[idx]
                    x, y = int(pt.x * w), int(pt.y * h)
                    image_points.append((x, y))
                
                image_points = np.array(image_points, dtype="double")
                
                # Camera calibration parameters
                focal_length = w
                center = (w // 2, h // 2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype="double")
                
                dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion
                
                # Calculate pose
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    self.model_points, image_points, camera_matrix, dist_coeffs)
                
                # Only draw visualization occasionally
                if self.frame_count % 6 == 0:
                    # Project a 3D point for visualization
                    (nose_end_point2D, _) = cv2.projectPoints(
                        np.array([(0.0, 0.0, 1000.0)]),
                        rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                    
                    p1 = tuple(image_points[0].astype(int))
                    p2 = tuple(nose_end_point2D[0][0].astype(int))
                    cv2.line(frame, p1, p2, (255, 0, 0), 3)
                
                # Calculate Euler angles
                rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
                proj_matrix = np.hstack((rvec_matrix, translation_vector))
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
                x_angle, y_angle, z_angle = euler_angles.flatten()
                
                # Display angles less frequently
                if self.frame_count % 5 == 0:
                    cv2.putText(frame, f"x: {x_angle:.2f}", (w - 200, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, f"y: {y_angle:.2f}", (w - 200, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Detect head position
                if y_angle < -20:
                    head_position = "Looking Left"
                elif y_angle > 20:
                    head_position = "Looking Right"
                elif x_angle < -20:
                    head_position = "Looking Up"
                elif x_angle > 20:
                    head_position = "Looking Down"
            except (IndexError, cv2.error):
                # Skip if pose estimation fails
                pass
                
        return head_position
    
    def update_status_overlay(self, frame, status_data):
        """Create a persistent overlay for status display"""
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Set up background for status area
        cv2.rectangle(overlay, (10, 10), (280, 180), (50, 50, 50), -1)
        cv2.rectangle(overlay, (10, 10), (280, 180), (100, 100, 100), 2)
        
        # Title bar
        cv2.rectangle(overlay, (10, 10), (280, 40), (70, 70, 70), -1)
        cv2.putText(overlay, "Driver Status", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add status items with priority
        y_offset = 70
        
        # Only show eyes closed alert if no higher priority alerts are active
        show_eyes_closed = status_data['eyes_closed'] and not (
            status_data['phone_use'] or 
            status_data['eating_drinking'] or 
            status_data['yawning']
        )
        
        if status_data['phone_use']:
            cv2.putText(overlay, "Phone Use", (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            y_offset += 30
        
        if status_data['eating_drinking']:
            cv2.putText(overlay, "Eating/Drinking", (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            y_offset += 30
        
        if status_data['yawning']:
            cv2.putText(overlay, "Yawning", (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
        
        if show_eyes_closed:
            cv2.putText(overlay, "Eyes Closed!", (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
        
        if status_data['head_position'] != "Forward":
            cv2.putText(overlay, f"Head: {status_data['head_position']}", (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add overall status at bottom of frame
        if status_data['distracted']:
            cv2.rectangle(overlay, (10, h-80), (450, h-20), (0, 0, 150), -1)
            cv2.rectangle(overlay, (10, h-80), (450, h-20), (0, 0, 255), 3)
            cv2.putText(overlay, "DRIVER DISTRACTED!", (20, h-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            cv2.rectangle(overlay, (10, h-80), (450, h-20), (0, 100, 0), -1)
            cv2.rectangle(overlay, (10, h-80), (450, h-20), (0, 255, 0), 3)
            cv2.putText(overlay, "Driver Attentive", (20, h-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # FPS counter
        cv2.rectangle(overlay, (w-150, h-50), (w-10, h-10), (40, 40, 40), -1)
        
        return overlay
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        # Set lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Start the YOLO thread
        self.start_yolo_thread()
        
        prev_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            h, w = frame.shape[:2]
            
            # Process with MediaPipe Holistic
            # Convert to RGB without copying the frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Holistic
            results = self.holistic.process(rgb)
            
            # Submit a frame for YOLO processing if the queue is empty
            current_yolo_results = None
            
            # Only run YOLO every few frames
            if self.frame_count % self.frame_skip == 0:
                if self.yolo_queue.empty():
                    # Make a copy to avoid modification during processing
                    self.yolo_queue.put(frame.copy(), block=False)
            
            # Check if there are any YOLO results available
            try:
                if not self.yolo_results_queue.empty():
                    self.last_yolo_results = self.yolo_results_queue.get(block=False)
                current_yolo_results = self.last_yolo_results
            except queue.Empty:
                pass
            
            # Extract ear and mouth coordinates
            ear_x, ear_y = -1, -1
            mouth_x, mouth_y = -1, -1
            
            if results.face_landmarks:
                try:
                    # Ear/cheek region
                    left_ear = results.face_landmarks.landmark[234]
                    right_ear = results.face_landmarks.landmark[454]
                    ear_x = int((left_ear.x + right_ear.x) / 2 * w)
                    ear_y = int((left_ear.y + right_ear.y) / 2 * h)
                    
                    # Only visualize occasionally
                    if self.frame_count % 5 == 0:
                        cv2.circle(frame, (ear_x, ear_y), 6, (0, 255, 0), -1)
                    
                    # Mouth center
                    mouth_top = results.face_landmarks.landmark[13]
                    mouth_bottom = results.face_landmarks.landmark[14]
                    mouth_x = int((mouth_top.x + mouth_bottom.x) / 2 * w)
                    mouth_y = int((mouth_top.y + mouth_bottom.y) / 2 * h)
                    
                    if self.frame_count % 5 == 0:
                        cv2.circle(frame, (mouth_x, mouth_y), 5, (0, 255, 0), -1)
                except IndexError:
                    # Skip if landmark indices aren't available
                    pass
            
            # Process all detection modules
            phone_detected, hand_near_ear = self.process_phone_detection(
                frame, results, (ear_x, ear_y), current_yolo_results)
            
            eating_detected, hand_to_mouth, mouth_opening = self.process_eating_drinking(
                frame, results, (mouth_x, mouth_y), current_yolo_results)
            
            eyes_closed, yawning = self.process_ear_mor(frame, results)
            
            head_position = self.process_head_position(frame, results)
            
            # Update detection states
            self.detections['phone_use'] = phone_detected or hand_near_ear
            self.detections['eating_drinking'] = eating_detected or hand_to_mouth
            self.detections['eyes_closed'] = eyes_closed
            self.detections['yawning'] = yawning
            self.detections['head_position'] = head_position
            
            # Determine if driver is distracted
            not_forward = head_position != "Forward"
            self.detections['distracted'] = (
                self.detections['phone_use'] or 
                self.detections['eating_drinking'] or 
                self.detections['eyes_closed'] or 
                self.detections['yawning'] or 
                not_forward
            )
            
            # Update status overlay - but only periodically to avoid flickering
            current_time = time.time()
            if current_time - self.last_status_update_time >= self.STATUS_UPDATE_INTERVAL:
                self.status_overlay = self.update_status_overlay(frame, self.detections)
                self.last_status_update_time = current_time
            
            # If we have a status overlay, blend it with the frame
            if self.status_overlay is not None:
                # Add status overlay with alpha blending
                alpha = 0.7  # Transparency factor
                mask = self.status_overlay.astype(bool)
                frame[mask] = cv2.addWeighted(frame, 0.3, self.status_overlay, 0.7, 0)[mask]
            
            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow("Driver Monitoring System", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break
        
        # Clean up
        self.yolo_thread_active = False
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    driver_monitor = DriverMonitoringSystem()
    driver_monitor.run()