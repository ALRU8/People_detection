import cv2
from .base_detect import BaseDetector


class WebcamDetector(BaseDetector):

    def __init__(self, camera_id=0, **kwargs):
        super().__init__(**kwargs)
        self.camera_id = camera_id

    def detect_frame(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise ValueError(f"camera {self.camera_id} not found")

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("couldn't get the shot")

        return self.detect(frame)

    def stream(self, callback=None, max_frames=None):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise ValueError(f"camera {self.camera_id} not found")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                det = self.detect(frame)
                if callback:
                    callback(frame, det, frame_idx)
                frame_idx += 1
                if max_frames and frame_idx >= max_frames:
                    break
        finally:
            cap.release()
