import cv2
import numpy as np
from ultralytics import YOLO


class BaseDetector:
    MODEL_PATH = "yolo26m.pt" #model
    CONFIDENCE = 0.35 #detection threshold
    IOU = 0.5 #NMS threshold
    IMGSZ = 1280 #input image size

    MIN_BOX_W = 12 #minimum width
    MIN_BOX_H = 28 #minimum height
    MIN_BOX_AREA_RATIO = 0.00008 #minimum frame area
    MIN_PERSON_ASPECT = 0.65 #minimum height/width ratio
    MAX_PERSON_ASPECT = 7.0 #maximum height/width ratio

    NMS_IOU = 0.5 #my NMS after YOLO
    SMOOTH_ALPHA = 0.6 #smoothing frames between frames

    def __init__(self, model_path=None, device=None, smooth=False):
        self.model_path = model_path or self.MODEL_PATH
        self.device = device
        self.model = None
        self.smooth = smooth
        self._prev_boxes = None

    def load_model(self):
        import torch

        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        try:
            self.model.fuse()
        except Exception:
            pass

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=640, conf=self.CONFIDENCE,
                           iou=self.IOU, classes=[0], device=self.device,
                           verbose=False)
        return self

    def _nms(self, detections):
        if len(detections) <= 1:
            return detections

        boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        scores = np.array([d["confidence"] for d in detections], dtype=np.float32)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            remaining = np.where(iou <= self.NMS_IOU)[0]
            order = order[remaining + 1]

        return [detections[i] for i in keep]

    def _smooth_boxes(self, detections):
        if self._prev_boxes is None or len(self._prev_boxes) != len(detections):
            self._prev_boxes = [None] * len(detections)
            return detections

        smoothed = []
        for i, det in enumerate(detections):
            new_box = det["bbox"]
            old_box = self._prev_boxes[i] if i < len(self._prev_boxes) else None

            if old_box is None:
                smoothed.append(det)
                self._prev_boxes[i] = new_box
                continue

            a = self.SMOOTH_ALPHA
            sx1 = int(old_box[0] * a + new_box[0] * (1 - a))
            sy1 = int(old_box[1] * a + new_box[1] * (1 - a))
            sx2 = int(old_box[2] * a + new_box[2] * (1 - a))
            sy2 = int(old_box[3] * a + new_box[3] * (1 - a))

            smoothed.append({"bbox": [sx1, sy1, sx2, sy2], "confidence": det["confidence"]})
            self._prev_boxes[i] = [sx1, sy1, sx2, sy2]

        return smoothed

    def clip_box(self, box, w, h):
        x1, y1, x2, y2 = box
        return (
            max(0, min(w - 1, int(x1))),
            max(0, min(h - 1, int(y1))),
            max(0, min(w - 1, int(x2))),
            max(0, min(h - 1, int(y2))),
        )

    def is_good_person_box(self, box, conf, frame_w, frame_h):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1

        if bw < self.MIN_BOX_W or bh < self.MIN_BOX_H:
            return False

        area = bw * bh
        if area / (frame_w * frame_h) < self.MIN_BOX_AREA_RATIO:
            return False

        aspect = bh / max(bw, 1)
        if not (self.MIN_PERSON_ASPECT <= aspect <= self.MAX_PERSON_ASPECT):
            return False

        if conf < self.CONFIDENCE:
            return False

        return True

    def detect(self, frame):
        if self.model is None:
            self.load_model()

        h, w = frame.shape[:2]
        results = self.model.predict(
            frame, conf=self.CONFIDENCE, iou=self.IOU,
            imgsz=self.IMGSZ, classes=[0], device=self.device, verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                clipped = self.clip_box((x1, y1, x2, y2), w, h)

                if self.is_good_person_box(clipped, conf, w, h):
                    detections.append({
                        "bbox": list(clipped),
                        "confidence": round(conf, 4),
                    })

        detections = self._nms(detections)

        if self.smooth:
            detections = self._smooth_boxes(detections)

        return {"count": len(detections), "detections": detections}

    def draw(self, frame, detections):
        for det in detections["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"person {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 1)
            y_top = max(0, y1 - th - 10)

            cv2.rectangle(frame, (x1, y_top), (x1 + tw + 6, y_top + th + 8), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 3, y_top + th + 3),
                        font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        return frame
