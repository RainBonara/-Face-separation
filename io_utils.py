"""Unicode-safe image read/write (cv2.imread/imwrite fail on non-ASCII Windows paths)."""

import os

import cv2
import numpy as np


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
