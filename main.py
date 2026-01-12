import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"


def ensure_model(path: str) -> str:
    """Download the face_landmarker task file if it's missing."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def main() -> None:
    model_path = ensure_model(MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )

    cap = cv2.VideoCapture(0)
    frame_idx = 0

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, frame_idx * 33)

                if result and result.face_landmarks:
                    for face in result.face_landmarks:
                        for lm in face:
                            x = int(lm.x * frame.shape[1])
                            y = int(lm.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_idx += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
