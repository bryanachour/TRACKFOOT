from pathlib import Path
from typing import List, Optional

import numpy as np
import supervision as sv
from ultralytics import YOLO


def _load_optimized(weights_path: Path, device: Optional[str], imgsz: int, half: bool) -> YOLO:
    """Load YOLO weights; auto-export to TensorRT .engine on first call if device=GPU
    and tensorrt is installed. Falls back to .pt with half precision otherwise."""
    weights_path = Path(weights_path)
    engine_path = weights_path.with_suffix(".engine")

    use_gpu = device is not None and device != "cpu"
    if use_gpu and engine_path.exists():
        try:
            return YOLO(str(engine_path), task="detect")
        except Exception:
            pass

    if use_gpu:
        try:
            import tensorrt  # noqa: F401
            pt_model = YOLO(str(weights_path))
            pt_model.export(format="engine", half=half, imgsz=imgsz, device=device, dynamic=False, batch=1, verbose=False)
            if engine_path.exists():
                return YOLO(str(engine_path), task="detect")
        except Exception:
            pass

    model = YOLO(str(weights_path))
    if use_gpu:
        try:
            model.to(f"cuda:{device}" if str(device).isdigit() else device)
        except Exception:
            pass
    return model


class YoloDetector:
    def __init__(self, weights_path: Path, conf: float = 0.30, iou: float = 0.50, device: Optional[str] = None, imgsz: int = 640, half: bool = True):
        self.model = _load_optimized(weights_path, device, imgsz, half)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.half = half

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)

    def predict_batch(self, frames: List[np.ndarray]) -> List[sv.Detections]:
        if not frames:
            return []
        results = self.model.predict(
            frames,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        return [sv.Detections.from_ultralytics(r) for r in results]


class BallDetector(YoloDetector):
    def __call__(self, frame: np.ndarray, slicer_slice_wh=(640, 640)) -> sv.Detections:
        def _callback(patch: np.ndarray) -> sv.Detections:
            result = self.model.predict(
                patch,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
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
