import cv2
import numpy as np

try:
    import mss
except ImportError:
    mss = None

from .base_detect import BaseDetector


class ScreenDetector(BaseDetector):

    def __init__(self, region=None, backend="mss", **kwargs):
        super().__init__(**kwargs)
        self.region = region
        self.backend = backend
        self._sct = None

    def _get_sct(self):
        if self._sct is None:
            if mss is None:
                raise ImportError("need the library mss: pip install mss")
            self._sct = mss.MSS()
        return self._sct

    def capture_screen(self, region=None):
        sct = self._get_sct()
        target = region or self.region
        if target is None:
            target = sct.monitors[1]

        img = np.array(sct.grab(target), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def detect_from_screen(self, region=None):
        frame = self.capture_screen(region)
        return self.detect(frame)

    def detect_region(self, left, top, width, height):
        region = {"left": left, "top": top, "width": width, "height": height}
        return self.detect_from_screen(region)

    def detect_full_screen(self, monitor_index=1):
        sct = self._get_sct()
        region = sct.monitors[monitor_index]
        return self.detect_from_screen(region)

    def close(self):
        if self._sct is not None:
            self._sct.close()
            self._sct = None
