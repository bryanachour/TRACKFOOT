from pathlib import Path
from typing import Optional

import numpy as np
import supervision as sv
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, weights_path: Path, conf: float = 0.30, iou: float = 0.50, device: Optional[str] = None, imgsz: int = 1280):
        self.model = YOLO(str(weights_path))
        if device is not None and device != "cpu":
            try:
                self.model.to(f"cuda:{device}" if device.isdigit() else device)
            except Exception:
                pass
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)


class BallDetector(YoloDetector):
    def __call__(self, frame: np.ndarray, slicer_slice_wh=(640, 640)) -> sv.Detections:
        def _callback(patch: np.ndarray) -> sv.Detections:
            result = self.model.predict(
                patch,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )[0]
            return sv.Detections.from_ultralytics(result)

        try:
            slicer = sv.InferenceSlicer(callback=_callback, slice_wh=slicer_slice_wh)
        except TypeError:
            slicer = sv.InferenceSlicer(callback=_callback)
        return slicer(frame)


def split_by_class(detections: sv.Detections, class_id: int) -> sv.Detections:
    mask = detections.class_id == class_id
    return detections[mask]
