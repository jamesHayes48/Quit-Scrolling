import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

# Open Camera
cap = cv2.VideoCapture(0)

mp_image = mp.Image
mp_image_format = mp.ImageFormat
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options, 
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

object_detector_options = python.BaseOptions(model_asset_path='object_detection.tflite')
options = vision.ObjectDetectorOptions(
    base_options=object_detector_options,
    category_denylist=['person'], 
    score_threshold=0.5
)
object_detector = vision.ObjectDetector.create_from_options(options)

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
        face_results = detector.detect(mp_img)
        object_results = object_detector.detect(mp_img)

        # Draw face landmarks for each face
        if face_results.face_landmarks:
            for face in face_results.face_landmarks:
                #draw_landmarks(frame, face)
                # Draw outline of left and right eye
                draw_specific_landmarks(frame, face, (255, 0, 0), 1, LEFT_EYE)
                draw_specific_landmarks(frame, face, (255, 0, 0), 1, RIGHT_EYE)

                right_iris_points = [face[i] for i in RIGHT_IRIS]
        
        if object_results.detections:
            bbox = object_results.detections[0].bounding_box
            start_point = int(bbox.origin_x), int(bbox.origin_y)
            end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
