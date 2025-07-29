import cv2
import mediapipe as mp
import numpy as np
import time
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
prev_time = 0

# 3D model points of key face features for pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (0.0, -330.0, -65.0),    # Chin
    (-225.0, 170.0, -135.0), # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),# Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

# Indices for landmarks to be used
landmark_ids = [1, 152, 263, 33, 287, 57]

while True:
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    image_h, image_w = image.shape[:2]
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        image_points = []
        for idx in landmark_ids:
            pt = face_landmarks.landmark[idx]
            x, y = int(pt.x * image_w), int(pt.y * image_h)
            image_points.append((x, y))

        image_points = np.array(image_points, dtype="double")

        focal_length = image_w
        center = (image_w // 2, image_h // 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)

        # Project a 3D point (0,0,1000.0) onto the image plane.
        (nose_end_point2D, _) = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]),
            rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        p1 = tuple(image_points[0].astype(int))
        p2 = tuple(nose_end_point2D[0][0].astype(int))
        cv2.line(image, p1, p2, (255, 0, 0), 3)  # Blue nose direction line

        # Display rotation vector values
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        x_angle, y_angle, z_angle = euler_angles.flatten()

        cv2.putText(image, f"x: {x_angle:.2f}", (image_w - 200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(image, f"y: {y_angle:.2f}", (image_w - 200, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(image, f"z: {z_angle:.2f}", (image_w - 200, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Detect pose state
        if y_angle < -20:
            text = "Looking left"
        elif y_angle > 20:
            text = "Looking Right"
        elif x_angle < -20:
            text = "Looking up"
        elif x_angle > 20:
            text = "Looking Down"
        else:
            text = "Forward"

        cv2.putText(image, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 3)

    # Draw facial mesh
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(image, f"FPS: {int(fps)}", (20, image_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Head Pose Estimation', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
