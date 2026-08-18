#!/usr/bin/env python3
"""Crop eyes, nose, and mouth from a face photo and save each as a separate image."""

import argparse
import os
import sys
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# Nose landmarks aren't grouped by mediapipe like eyes/lips are, so we
# hand-pick points spanning the bridge, tip, and both alae/nostrils.
NOSE_INDICES = [5, 4, 1, 19, 94, 2, 129, 358, 48, 278]

FLC = vision.FaceLandmarksConnections


def _connection_indices(connections):
    return {i for c in connections for i in (c.start, c.end)}


REGIONS = {
    "left_eye": _connection_indices(FLC.FACE_LANDMARKS_LEFT_EYE),
    "right_eye": _connection_indices(FLC.FACE_LANDMARKS_RIGHT_EYE),
    "nose": set(NOSE_INDICES),
    "mouth": _connection_indices(FLC.FACE_LANDMARKS_LIPS),
}


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading face landmark model (one-time, ~4MB)...", file=sys.stderr)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def imread_unicode(path):
    # cv2.imread() can't open paths containing non-ASCII characters (e.g. Korean)
    # on Windows, since it opens the file using the system's active code page.
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image):
    # Same limitation as imread_unicode, but for writing.
    ext = os.path.splitext(path)[1]
    success, buf = cv2.imencode(ext, image)
    if success:
        buf.tofile(path)
    return success


def compute_bbox(image, landmarks, indices, padding):
    h, w = image.shape[:2]
    xs = [landmarks[i].x * w for i in indices]
    ys = [landmarks[i].y * h for i in indices]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    box_w = x_max - x_min
    box_h = y_max - y_min
    x_min -= box_w * padding
    x_max += box_w * padding
    y_min -= box_h * padding
    y_max += box_h * padding

    x_min = max(0, int(round(x_min)))
    y_min = max(0, int(round(y_min)))
    x_max = min(w, int(round(x_max)))
    y_max = min(h, int(round(y_max)))

    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def crop_region(image, bbox):
    if bbox is None:
        return None
    x_min, y_min, x_max, y_max = bbox
    return image[y_min:y_max, x_min:x_max]


def build_parts_only_image(image, bboxes):
    """Original-size RGBA image, transparent except inside the given region boxes."""
    h, w = image.shape[:2]
    result = np.zeros((h, w, 4), dtype=np.uint8)
    for bbox in bboxes:
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        region = image[y_min:y_max, x_min:x_max]
        alpha = np.full(region.shape[:2] + (1,), 255, dtype=np.uint8)
        result[y_min:y_max, x_min:x_max] = np.concatenate([region, alpha], axis=2)
    return result


def build_collage(crops, gap=10, background=255):
    """Stack eyes (side by side) on top, nose below, mouth at the bottom."""
    left_eye = crops.get("left_eye")
    right_eye = crops.get("right_eye")

    def pad_to_height(img, height):
        top = (height - img.shape[0]) // 2
        bottom = height - img.shape[0] - top
        return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(background,) * 3)

    eye_row = None
    if left_eye is not None and right_eye is not None:
        height = max(left_eye.shape[0], right_eye.shape[0])
        gap_col = np.full((height, gap, 3), background, dtype=np.uint8)
        eye_row = np.hstack([pad_to_height(left_eye, height), gap_col, pad_to_height(right_eye, height)])
    elif left_eye is not None:
        eye_row = left_eye
    elif right_eye is not None:
        eye_row = right_eye

    rows = [r for r in (eye_row, crops.get("nose"), crops.get("mouth")) if r is not None]
    if not rows:
        return None

    width = max(r.shape[1] for r in rows)

    def pad_to_width(img, target_width):
        left = (target_width - img.shape[1]) // 2
        right = target_width - img.shape[1] - left
        return cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(background,) * 3)

    gap_row = np.full((gap, width, 3), background, dtype=np.uint8)
    stacked = [pad_to_width(rows[0], width)]
    for row in rows[1:]:
        stacked.append(gap_row)
        stacked.append(pad_to_width(row, width))

    return np.vstack(stacked)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_image_paths(inputs):
    paths = []
    for path in inputs:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    paths.append(os.path.join(path, name))
        else:
            paths.append(path)
    return paths


def create_landmarker(max_faces):
    ensure_model()
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=max_faces,
    )
    return vision.FaceLandmarker.create_from_options(options)


def process_image(input_path, output_dir, padding, landmarker):
    image = imread_unicode(input_path)
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        print(f"No face detected in {input_path}", file=sys.stderr)
        return []

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []
    for face_idx, landmarks in enumerate(result.face_landmarks):
        suffix = f"_face{face_idx}" if len(result.face_landmarks) > 1 else ""

        bboxes = {
            region_name: compute_bbox(image, landmarks, indices, padding)
            for region_name, indices in REGIONS.items()
        }

        crops = {}
        for region_name, bbox in bboxes.items():
            crop = crop_region(image, bbox)
            if crop is None:
                continue
            crops[region_name] = crop
            out_path = os.path.join(output_dir, f"{base_name}{suffix}_{region_name}.png")
            imwrite_unicode(out_path, crop)
            saved_paths.append(out_path)

        parts_only = build_parts_only_image(image, bboxes.values())
        out_path = os.path.join(output_dir, f"{base_name}{suffix}_parts_only.png")
        imwrite_unicode(out_path, parts_only)
        saved_paths.append(out_path)

        collage = build_collage(crops)
        if collage is not None:
            out_path = os.path.join(output_dir, f"{base_name}{suffix}_collage.png")
            imwrite_unicode(out_path, collage)
            saved_paths.append(out_path)

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="+",
        help="Path(s) to input face photo(s), and/or folders containing photos",
    )
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")
    parser.add_argument(
        "--padding",
        type=float,
        default=0.3,
        help="Fractional padding added around each cropped region (default: 0.3)",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=1,
        help="Maximum number of faces to process in each image (default: 1)",
    )
    args = parser.parse_args()

    image_paths = collect_image_paths(args.input)
    if not image_paths:
        print("No image files found for the given input.", file=sys.stderr)
        sys.exit(1)

    landmarker = create_landmarker(args.max_faces)

    total_saved = 0
    for image_path in image_paths:
        try:
            saved = process_image(image_path, args.output, args.padding, landmarker)
        except ValueError as e:
            print(e, file=sys.stderr)
            continue
        if saved:
            print(f"Saved {len(saved)} image(s) from {image_path}:")
            for path in saved:
                print(f"  {path}")
            total_saved += len(saved)

    if total_saved == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
