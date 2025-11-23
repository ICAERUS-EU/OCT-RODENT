from __future__ import annotations

from typing import Dict, Optional, List, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
except Exception as _ex:  # pragma: no cover
    YOLO = None  # type: ignore

from ..pipeline import Frame, Plugin


class YOLOPlugin(Plugin):
    """Ultralytics YOLO inference plugin.

    Args:
        model_path: path to .pt model file.
        conf: confidence threshold.
        device: 'cpu' or 'cuda'.
        imgsz: optional input size; if None, YOLO decides.
        draw: draw detections onto frame.bgr.
        class_names: optional mapping for class ids to names for labels.
    """

    def __init__(
        self,
        model_path: str,
        *,
        conf: float = 0.5,
        device: str = 'cpu',
        imgsz: Optional[int] = None,
        draw: bool = True,
        class_names: Optional[Dict[int, str]] = None,
        nms_iou: float = 0.45,
        # Multi-scale options
        multi_scale: bool = False,
        ms_scales: Optional[List[float]] = None,
        ms_score_thresh: float = 0.25,
        ms_iou_thresh: float = 0.45,
        ms_per_class_nms: bool = True        
    ) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed; cannot use YOLOPlugin")
        self.name = "yolo"
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.imgsz = imgsz
        self.draw = draw
        # Resolve class names: prefer provided mapping; else try to read from model
        resolved_names: Dict[int, str] = {}
        if class_names and isinstance(class_names, dict) and len(class_names) > 0:
            try:
                resolved_names = {int(k): str(v) for k, v in class_names.items()}
            except Exception:
                resolved_names = {}
        if not resolved_names:
            # Try ultralytics model's names attribute
            try:
                names_attr = getattr(self.model, 'names', None)
                if names_attr is None and hasattr(self.model, 'model'):
                    names_attr = getattr(self.model.model, 'names', None)
                if isinstance(names_attr, dict):
                    resolved_names = {int(k): str(v) for k, v in names_attr.items()}
                elif isinstance(names_attr, (list, tuple)):
                    resolved_names = {int(i): str(v) for i, v in enumerate(names_attr)}
            except Exception:
                resolved_names = {}
        self.class_names = resolved_names
        # Multi-scale config
        self.multi_scale = bool(multi_scale)
        self.ms_scales = ms_scales if (ms_scales and len(ms_scales) > 0) else [1.0]
        self.ms_score_thresh = float(ms_score_thresh)
        self.ms_iou_thresh = float(ms_iou_thresh)
        self.ms_per_class_nms = bool(ms_per_class_nms)
        self.nms_iou = float(nms_iou)

    def process(self, frame: Frame, ctx: Dict[str, object]) -> None:
        dets: List[Tuple[int, int, int, int, float, int]] = []  # (x1,y1,x2,y2,conf,cls)

        if not self.multi_scale or (self.ms_scales == [1.0]):
            # Single-scale inference (original behavior)
            kwargs = dict(conf=self.conf, device=self.device, verbose=False)
            if self.nms_iou is not None:
                kwargs['iou'] = self.nms_iou
            if self.imgsz is not None:
                kwargs['imgsz'] = self.imgsz
            try:
                results = self.model.predict(source=frame.bgr, **kwargs)
            except TypeError:
                results = self.model.predict(frame.bgr, conf=self.conf)

            if results and len(results) > 0:
                r = results[0]
                boxes = getattr(r, 'boxes', None)
                if boxes is not None:
                    try:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else np.zeros((xyxy.shape[0],), dtype=np.float32)
                        clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else np.zeros((xyxy.shape[0],), dtype=int)
                        for (x1, y1, x2, y2), c, cl in zip(xyxy, confs, clss):
                            dets.append((int(x1), int(y1), int(x2), int(y2), float(c), int(cl)))
                    except Exception:
                        pass
        else:
            # Multi-scale inference
            all_boxes = []
            all_scores = []
            all_classes = []

            for s in self.ms_scales:
                if s <= 0:
                    continue
                if abs(s - 1.0) < 1e-6:
                    img_s = frame.bgr
                else:
                    h, w = frame.bgr.shape[:2]
                    new_w = max(1, int(round(w * s)))
                    new_h = max(1, int(round(h * s)))
                    img_s = cv2.resize(frame.bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                kwargs = dict(conf=self.conf, device=self.device, verbose=False)
                if self.imgsz is not None:
                    kwargs['imgsz'] = self.imgsz
                try:
                    results = self.model.predict(source=img_s, **kwargs)
                except TypeError:
                    results = self.model.predict(img_s, conf=self.conf)
                if not results or len(results) == 0:
                    continue

                r = results[0]
                boxes = getattr(r, 'boxes', None)
                if boxes is None or len(boxes) == 0:
                    continue
                try:
                    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
                    scores = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else np.zeros((xyxy.shape[0],), dtype=np.float32)
                    clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else np.zeros((xyxy.shape[0],), dtype=int)
                    # Map back to original coordinates by dividing by scale
                    if abs(s - 1.0) > 1e-6:
                        xyxy[:, [0, 2]] /= float(s)
                        xyxy[:, [1, 3]] /= float(s)
                    all_boxes.append(xyxy)
                    all_scores.append(scores)
                    all_classes.append(clss)
                except Exception:
                    continue

            if len(all_boxes) > 0:
                boxes = np.vstack(all_boxes)
                scores = np.concatenate(all_scores)
                classes = np.concatenate(all_classes)

                # Score threshold
                keep = scores >= self.ms_score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                classes = classes[keep]

                if boxes.shape[0] > 0:
                    # NMS
                    if self.ms_per_class_nms:
                        final_idx: List[int] = []
                        for cls in np.unique(classes):
                            idxs = np.where(classes == cls)[0]
                            if idxs.size == 0:
                                continue
                            kept = self._nms_numpy(boxes[idxs], scores[idxs], self.ms_iou_thresh)
                            final_idx.extend(idxs[kept].tolist())
                        final_idx = np.array(sorted(final_idx), dtype=int)
                    else:
                        final_idx = self._nms_numpy(boxes, scores, self.ms_iou_thresh)

                    boxes = boxes[final_idx]
                    scores = scores[final_idx]
                    classes = classes[final_idx]

                    for (x1, y1, x2, y2), sc, cl in zip(boxes, scores, classes):
                        dets.append((int(x1), int(y1), int(x2), int(y2), float(sc), int(cl)))

        ctx[self.name] = dets

        if self.draw:
            for (x1, y1, x2, y2, c, cl) in dets:
                cv2.rectangle(frame.bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
                label = f"{self.class_names.get(cl, str(cl))}:{c:.2f}"
                org = (x1, max(12, y1 - 4))
                # outline for readability
                cv2.putText(frame.bgr, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2, cv2.LINE_8)
                cv2.putText(frame.bgr, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_8)

    # --- Utils ---
    @staticmethod
    def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
        """Simple NMS in numpy. boxes: (N,4) xyxy, scores: (N,). Returns kept indices (np.ndarray[int])."""
        if boxes.shape[0] == 0:
            return np.array([], dtype=int)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            denom = (areas[i] + areas[order[1:]] - inter + 1e-8)
            iou = inter / denom
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=int)
