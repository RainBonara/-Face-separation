"""Which face landmarks make up each region, and how to turn them into crop boxes."""

from mediapipe.tasks.python import vision

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
