# Distracted Driver Detection

This project focuses on identifying distracted driver behaviors using computer vision and deep learning techniques. It detects actions such as drowsiness, yawning, phone usage, eating, drinking, and unusual head orientation — all of which can lead to dangerous driving conditions.

Due to the limitations of the available dataset and the challenges of implementing a dynamic real-time environment, we extended our approach by integrating multiple libraries such as YOLOv8, MediaPipe, and OpenCV to improve detection accuracy and reliability.

---

## Modules and Techniques

### 1. Head Pose Estimation
- **Technology**: MediaPipe Face Mesh + 3D PnP
- **Purpose**: Detect head orientation (e.g., looking down or sideways)

### 2. Drowsiness Detection
- **Technology**: MediaPipe Face Mesh
- **Purpose**: Identify fatigue or micro-sleep events based on Eye Aspect Ratio (EAR)

### 3. Yawning Detection
- **Technology**: MediaPipe Face Mesh
- **Purpose**: Detect sustained mouth opening as a sign of sleepiness

### 4. Eating Detection
- **Technology**: YOLOv8 + MediaPipe Hands
- **Purpose**: Detect eating gestures using object and hand proximity

### 5. Drinking Detection
- **Technology**: YOLOv8 + MediaPipe Hands
- **Purpose**: Detect if a person is drinking while driving

### 6. Phone/Calling Detection
- **Technology**: YOLOv8 + MediaPipe Hands + Pose
- **Purpose**: Detect phone usage through object detection and hand-to-ear gesture

### 7. Gesture and Proximity Analysis
- **Technology**: MediaPipe Hands & Face
- **Purpose**: Measure distances and angles between hands and key face landmarks

### 8. Output Layer (GUI and Alerts)
- **Technology**: OpenCV
- **Purpose**: Real-time feedback using visual overlays and alerts

---

## Technology Stack

- Python  
- OpenCV  
- MediaPipe  
- YOLOv8 (Ultralytics)  
- NumPy, SciPy, Matplotlib  
- Jupyter Notebook

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Distracted-Driver-Detection.git
   cd Distracted-Driver-Detection
Install the dependencies:


To run all components together, use:

python main2.py
Note: Running all modules simultaneously may cause lag or performance issues due to high computational load.

To test individual components (recommended for analysis or debugging), run the specific files like:

python head_position.py
python EAT_DRINK.py
# and so on...
Limitations
The dataset used was limited and not suitable for generalizing across all real-world scenarios.

Single-model solutions were insufficient, so we used a modular approach combining YOLO, MediaPipe, and OpenCV.

Real-time performance may be affected when multiple modules run together.

