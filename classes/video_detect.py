import cv2
from .base_detect import BaseDetector


class VideoDetector(BaseDetector):

    def detect_from_path(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"couldn't open: {video_path}")

        results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            det = self.detect(frame)
            results.append({"frame": frame_idx, **det})
            frame_idx += 1

        cap.release()
        return {"total_frames": frame_idx, "results": results}

    def detect_frame(self, video_path, frame_number):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"couldn't open: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"shot {frame_number} not read")

        return self.detect(frame)

    def get_video_info(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"couldn't open: {video_path}")

        info = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        cap.release()
        return info
