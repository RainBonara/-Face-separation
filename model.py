"""Face landmark model download and MediaPipe FaceLandmarker setup."""

import os
import sys
import urllib.request

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading face landmark model (one-time, ~4MB)...", file=sys.stderr)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_landmarker(max_faces):
    ensure_model()
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=max_faces,
    )
    return vision.FaceLandmarker.create_from_options(options)
