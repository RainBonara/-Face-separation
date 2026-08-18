"""Combine cropped regions into the parts-only and collage output images."""

import cv2
import numpy as np


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
