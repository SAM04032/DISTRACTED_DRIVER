import cv2
import pygame
import imutils
from scipy.spatial import distance as dist
import numpy as np
import time

# Initialize pygame mixer for alert sound
pygame.mixer.init()
try:
    pygame.mixer.music.load("./music.wav")
    print("Sound file loaded successfully")
except:
    print("Could not load 'music.wav'. Make sure it exists in the same directory.")
    exit()

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ADJUSTED PARAMETERS FOR BETTER DETECTION
EAR_THRESHOLD = 0.3   # Increased from 0.25 to make more sensitive to eye closure
CONSEC_FRAMES = 10    # Reduced frames needed to trigger alert
counter = 0
alert_displayed = False

# Use Haar cascade for face detection and alternative eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

# Additional eye detector as backup
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Manual test mode - for detecting if the system works
MANUAL_TEST_MODE = True
test_timer = 0
debug_text = ""

# Use webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Webcam not accessible")
    exit()

print("Distracted Driver Detection Started - Press ESC to exit")
print("Sound will play when eyes close")
print("Debug mode enabled: Press 'T' to trigger a test alert")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Frame not captured.")
        continue

    # Resize the frame
    frame = imutils.resize(frame, width=450)
    
    # Add debug mode info
    key = cv2.waitKey(1) & 0xFF
    
    # Test sound manually with T key
    if MANUAL_TEST_MODE and key == ord('t'):
        print("Manual test triggered!")
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1)
            test_timer = 100  # Will play for about 3 seconds
            debug_text = "MANUAL TEST: PLAYING SOUND"
    
    # Count down test timer
    if test_timer > 0:
        test_timer -= 1
        if test_timer == 0:
            pygame.mixer.music.stop()
            debug_text = ""
    
    # Exit with ESC
    if key == 27:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        break
    
    # Real detection logic starts here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # If no faces are detected, reset
    if len(faces) == 0:
        counter = 0
        if pygame.mixer.music.get_busy() and test_timer == 0:
            pygame.mixer.music.stop()
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Process each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Region of interest for face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Define upper half of face for eye detection
        upper_half_y = y + h//4
        upper_half_h = h//2
        
        # Try to detect eyes using specialized eye cascades first
        left_eye = left_eye_cascade.detectMultiScale(
            roi_gray[0:upper_half_h, 0:w//2],  # Left half of upper face
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        right_eye = right_eye_cascade.detectMultiScale(
            roi_gray[0:upper_half_h, w//2:w],  # Right half of upper face
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Fall back to general eye detector if needed
        # FIXED: Initialize as a list and keep it as a list
        eyes = []
        if len(left_eye) == 0 or len(right_eye) == 0:
            detected_eyes = eye_cascade.detectMultiScale(
                roi_gray[0:upper_half_h, :],
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            # Add detected eyes to our list
            for (ex, ey, ew, eh) in detected_eyes:
                eyes.append((ex, ey, ew, eh))
        
        # Add left eye detections to eyes list
        for (ex, ey, ew, eh) in left_eye:
            eyes.append((ex, ey, ew, eh))
        
        # Add right eye detections to eyes list
        for (ex, ey, ew, eh) in right_eye:
            eyes.append((ex + w//2, ey, ew, eh))  # Adjust x-position
        
        # Draw all detected eyes and calculate EAR
        total_ear = 0
        num_eyes = 0
        
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Calculate simple EAR based on rectangle dimensions
            # For closed eyes, height (eh) will be much smaller relative to width (ew)
            simple_ear = eh / ew if ew > 0 else 1.0
            total_ear += simple_ear
            num_eyes += 1
        
        # Handle cases where eyes are detected
        if num_eyes >= 1:
            avg_ear = total_ear / num_eyes
            
            # Display EAR value
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (300, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Check if eyes appear closed
            if avg_ear < EAR_THRESHOLD:
                counter += 1
                cv2.putText(frame, f"Eyes Closing! {counter}/{CONSEC_FRAMES}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Trigger alert if eyes closed for enough frames
                if counter >= CONSEC_FRAMES:
                    if not pygame.mixer.music.get_busy() or test_timer > 0:
                        pygame.mixer.music.play(-1)
                        print("ALERT! Eyes closed detected!")
                    
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                # Reset when eyes are open
                counter = 0
                if pygame.mixer.music.get_busy() and test_timer == 0:
                    pygame.mixer.music.stop()
                    print("Eyes open detected")
        else:
            # If no eyes detected, consider if it's because eyes are closed
            counter += 1
            cv2.putText(frame, f"Eyes not detected! {counter}/{CONSEC_FRAMES}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if counter >= CONSEC_FRAMES:
                if not pygame.mixer.music.get_busy() or test_timer > 0:
                    pygame.mixer.music.play(-1)
                    print("ALERT! No eyes detected for several frames!")
                
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Display manual test message if active
    if debug_text:
        cv2.putText(frame, debug_text, (10, 90),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    # Display instructions
    cv2.putText(frame, "Press 'T' to test sound", (10, frame.shape[0] - 10),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Drowsiness Detection", frame)

cap.release()
cv2.destroyAllWindows()
