"""Resolve CLI input arguments (files and/or folders) into a list of image paths."""

import os

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
