import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LEFT_EYE = [33, 133, 160, 158, 153, 144, 163, 7]
RIGHT_EYE = [362, 263, 387, 385, 380, 373, 390, 249]

LEFT_CHEEK = 234
RIGHT_CHEEK = 454
NOSE = 1

YAW_THRESHOLD = 0.06

def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=1):
    '''Draw landmarks on face detected
    
    :param image: image to draw on
    :param landmarks: list of landmarks
    :param color: color of the dots
    :param radius: radius of the dots
    '''
    
    for lm in landmarks:
        x = int(lm.x * image.shape[1])
        y = int(lm.y * image.shape[0])
        cv2.circle(image, (x, y), radius, color, -1)

def draw_specific_landmarks(image, landmarks, color=(255, 0, 0), radius=1, indices=[]):
    '''Draw specific landmarks on face detected based on given indices
    
    :param image: image to draw on
    :param landmarks: list of landmarks
    :param color: color of the dots
    :param radius: radius of the dots
    :param indices: list of landmark indices to draw
    '''
    
    for idx in indices:
        lm = landmarks[idx]
        x = int(lm.x * image.shape[1])
        y = int(lm.y * image.shape[0])
        cv2.circle(image, (x, y), radius, color, -1)


def get_face_yaw_score(landmarks):
    left_x = landmarks[LEFT_CHEEK].x
    right_x = landmarks[RIGHT_CHEEK].x
    nose_x = landmarks[NOSE].x

    cheek_mid_x = (left_x + right_x) / 2.0
    face_width = abs(right_x - left_x)
    if face_width < 1e-6:
        return 0.0

    # Normalized horizontal nose offset relative to face width.
    return (nose_x - cheek_mid_x) / face_width


def get_face_direction(yaw_score):
    if yaw_score > YAW_THRESHOLD:
        return 'LEFT'
    if yaw_score < -YAW_THRESHOLD:
        return 'RIGHT'
    return 'CENTER'


# Open Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
mp_image = mp.Image
mp_image_format = mp.ImageFormat
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options, 
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

duration_window = 5
no_object_start_time = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image(image_format=mp_image_format.SRGB, data=rgb)
        face_results = detector.detect(mp_img)

        has_face = bool(face_results.face_landmarks)
       
        # Draw face landmarks for each face
        if face_results.face_landmarks:
            for face in face_results.face_landmarks:
                #draw_landmarks(frame, face)
                yaw_score = get_face_yaw_score(face)
                direction = get_face_direction(yaw_score)
                cv2.putText(
                    frame,
                    f'Facing: {direction} ({yaw_score:+.3f})',
                    (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )

        # Start and maintain timer only while a face is present and object is missing.
        if has_face and not has_object:
            if no_object_start_time is None:
                no_object_start_time = time.perf_counter()

            elapsed_time = time.perf_counter() - no_object_start_time
            cv2.putText(
                frame,
                f'Face w/o object: {elapsed_time:.1f}s',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 165, 255),
                2,
            )

            if elapsed_time >= duration_window:
                cv2.putText(frame, 'Object missing too long', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            no_object_start_time = None


        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
