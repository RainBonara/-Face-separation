#!/usr/bin/env python3
"""Crop eyes, nose, and mouth from a face photo and save each as a separate image."""

import argparse
import os
import sys

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Nose landmarks aren't grouped by mediapipe like eyes/lips are, so we
# hand-pick points spanning the bridge, tip, and both alae/nostrils.
NOSE_INDICES = [5, 4, 1, 19, 94, 2, 129, 358, 48, 278]

REGIONS = {
    "left_eye": lambda mesh: {i for pair in mesh.FACEMESH_LEFT_EYE for i in pair},
    "right_eye": lambda mesh: {i for pair in mesh.FACEMESH_RIGHT_EYE for i in pair},
    "nose": lambda mesh: set(NOSE_INDICES),
    "mouth": lambda mesh: {i for pair in mesh.FACEMESH_LIPS for i in pair},
}


def crop_region(image, landmarks, indices, padding):
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
    return image[y_min:y_max, x_min:x_max]


def process_image(input_path, output_dir, padding, max_faces):
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_faces,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        print(f"No face detected in {input_path}", file=sys.stderr)
        return []

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []
    for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
        landmarks = face_landmarks.landmark
        suffix = f"_face{face_idx}" if len(results.multi_face_landmarks) > 1 else ""

        for region_name, index_fn in REGIONS.items():
            indices = index_fn(mp_face_mesh)
            crop = crop_region(image, landmarks, indices, padding)
            if crop is None:
                continue
            out_path = os.path.join(output_dir, f"{base_name}{suffix}_{region_name}.png")
            cv2.imwrite(out_path, crop)
            saved_paths.append(out_path)

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to an input face photo")
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
        help="Maximum number of faces to process in the image (default: 1)",
    )
    args = parser.parse_args()

    saved = process_image(args.input, args.output, args.padding, args.max_faces)
    if not saved:
        sys.exit(1)

    print(f"Saved {len(saved)} image(s) to {args.output}:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
