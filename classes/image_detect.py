import cv2
import numpy as np
from .base_detect import BaseDetector


class ImageDetector(BaseDetector):

    def detect_from_path(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"couldn't download: {image_path}")
        return self.detect(frame)

    def detect_from_bytes(self, image_bytes):
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("image could not be decoded")
        return self.detect(frame)

    def detect_from_array(self, frame):
        return self.detect(frame)
