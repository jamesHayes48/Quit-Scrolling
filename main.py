import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=1):
    '''Draw landmarks on face detected'''
    for lm in landmarks:
        x = int(lm.x * image.shape[1])
        y = int(lm.y * image.shape[0])
        cv2.circle(image, (x, y), radius, color, -1)

def draw_eye_outline(image, landmarks, indices):
    '''Draw eye outline using given landmark indices'''
    points = []
    for idx in indices:
        lm = landmarks[idx]
        x = int(lm.x * image.shape[1])
        y = int(lm.y * image.shape[0])
        points.append((x, y))
    points = np.array(points, dtype=np.int32)
    cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)

# Open Camera
cap = cv2.VideoCapture(0)

mp_image = mp.Image
mp_image_format = mp.ImageFormat
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE = [33, 133, 160, 158, 153, 144, 163, 7]
RIGHT_EYE = [362, 263, 387, 385, 380, 373, 390, 249]

LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image(image_format=mp_image_format.SRGB, data=rgb)
        results = detector.detect(mp_img)

        # Draw face landmarks for each face
        if results.face_landmarks:
            for face in results.face_landmarks:
                draw_landmarks(frame, face)
                draw_eye_outline(frame, face, LEFT_EYE)
                draw_eye_outline(frame, face, RIGHT_EYE)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
