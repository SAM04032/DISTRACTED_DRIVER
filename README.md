# 🚗 Distracted Driver Detection System

This project aims to detect unsafe driving behaviors using computer vision and deep learning techniques. The system identifies various forms of driver distraction such as drowsiness, yawning, mobile phone usage, eating, drinking, and unusual head orientation — key indicators of risky driving conditions.

Due to dataset constraints and real-time performance requirements, we adopted a modular approach using powerful libraries like **YOLOv8**, **MediaPipe**, and **OpenCV** to achieve high detection reliability and accuracy.

---

## 🧠 Core Modules & Techniques

### 1. Head Pose Estimation  
- **Technology**: MediaPipe Face Mesh + PnP (Perspective-n-Point)  
- **Purpose**: Determine if the driver is looking away from the road.

### 2. Drowsiness Detection  
- **Technology**: MediaPipe Face Mesh  
- **Purpose**: Monitor eye aspect ratio (EAR) to identify signs of fatigue.

### 3. Yawning Detection  
- **Technology**: MediaPipe Face Mesh  
- **Purpose**: Detect prolonged mouth openings that indicate yawning.

### 4. Eating Detection  
- **Technology**: YOLOv8 + MediaPipe Hands  
- **Purpose**: Identify hand-to-mouth eating gestures using object detection.

### 5. Drinking Detection  
- **Technology**: YOLOv8 + MediaPipe Hands  
- **Purpose**: Recognize when a person is drinking from a bottle or cup.

### 6. Phone/Calling Detection  
- **Technology**: YOLOv8 + MediaPipe Hands + Pose  
- **Purpose**: Detect phone use by observing object detection and hand-to-ear posture.

### 7. Proximity & Gesture Analysis  
- **Technology**: MediaPipe Hands & Face  
- **Purpose**: Analyze distances and angles between hands and key facial landmarks.

### 8. Real-Time Feedback & Alerts  
- **Technology**: OpenCV  
- **Purpose**: Provide visual overlays and sound alerts in real-time.

---

## 🛠 Technology Stack

- Python  
- OpenCV  
- MediaPipe  
- YOLOv8 (Ultralytics)  
- NumPy, SciPy, Matplotlib  
- Jupyter Notebook

---

## 🚀 How to Run the Project

###  Clone the Repository



