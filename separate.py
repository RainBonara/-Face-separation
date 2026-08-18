#!/usr/bin/env python3
"""Crop eyes, nose, and mouth from a face photo and save each as a separate image."""

import argparse
import os
import sys

import cv2
import mediapipe as mp

from batch import collect_image_paths
from compose import build_collage, build_parts_only_image
from io_utils import imread_unicode, imwrite_unicode
from model import create_landmarker
from regions import REGIONS, compute_bbox, crop_region


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
