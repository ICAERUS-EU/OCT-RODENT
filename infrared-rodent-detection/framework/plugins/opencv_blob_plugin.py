from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import cv2
import numpy as np

from ..pipeline import Frame, Plugin


class MedianBackgroundBlobPlugin(Plugin):
    """Motion/blob detector with Celsius preference and grayscale fallback.

    Pipeline per frame:
      1) Maintain rolling median backgrounds for Celsius (if available) and grayscale.
      2) Optional Gaussian smoothing (noise_kernel) applied before differencing in both domains.
      3) Motion mask:
         - Celsius domain if celsius present & background ready: |C - bgC| >= motion_delta_c
         - Else grayscale: |gray - bgGray| >= motion_delta_gray (derived automatically from motion_delta_c if not provided)
      4) Relative (temporal) mask (rel_mode != off):
         - Celsius: hot (C-bgC >= delta_c), cold (C-bgC <= -delta_c), both (|C-bgC| >= delta_c)
         - Fallback grayscale: |gray - bgGray| >= delta_gray (derived from delta_c)
      5) Spatial hot-spot mask (warm static objects): current - blur(current) >= hot_delta_c (Celsius) OR >= hot_delta_gray (grayscale fallback).
      6) Combine motion + relative per combine (intersect/union); derive moving vs static (hot only, non-moving) separation.
      7) Morphology, contour extraction, filtering, tracking/persistence.

    Automatic grayscale threshold derivation rules (no CLI args):
      delta_gray = round(delta_c * 6)  # rough scale from typical 0.04 C per raw unit & 8-bit stretch
      motion_delta_gray = max(1, round(motion_delta_c * 6))
      hot_delta_gray = max(1, round(hot_delta_c * 6))
    """

    def __init__(
        self,
        *,
        buffer_size: int = 25,
        min_area: int = 100,
        kernel_open: int = 3,
        kernel_close: int = 5,
        dilate_iter: int = 1,
        draw: bool = True,
        label: str = "blob",
        # relative-to-background options
        rel_mode: str = "off",  # off|hot|cold|both
        delta_c: float = 1.0,    # degrees Celsius difference
        combine: str = "intersect",  # union|intersect with motion mask (default intersect to avoid cold-only motion)
        motion_delta_c: float = 0.5,
        # spatial hot-spot detection to catch warm static objects
        hot_static: bool = True,
        hot_kernel: int = 15,          # odd size for blur kernel
        hot_delta_c: float = 1.0,
        # area filtering (pixels)
        max_area: int = 0,  # reject blobs larger than this many pixels (<=0 disables)
        static_min_area: int = 0,
        static_min_area_mult: float = 1.0,
        # noise suppression
        noise_kernel: int = 5,
        # avoid absorbing detections into background
        bg_freeze_on_detect: bool = True,
        # persistence filtering (only keep long-lived objects)
        persist_min_frames: int = 10,
        persist_min_seconds: float = 0.0,
        track_max_miss: int = 5,
        track_iou_thresh: float = 0.3,
        # convert moving -> static after N consecutive static frames for that track
        static_after_frames: int = 5,
        # convert static -> moving after M consecutive moving frames (hysteresis)
        moving_after_frames: int = 2,
        # geometric stability criteria to consider an object static even if motion mask flickers
        static_pos_px: int = 3,           # center movement in pixels allowed between frames
        static_area_frac: float = 0.15,   # max relative area change between frames (e.g., 0.15 = 15%)
        static_iou: float = 0.85,         # alternatively, if IoU(prev,cur) >= this, treat as stable
        # moving vs static disambiguation based on mask coverage within box
        moving_coverage_max: float = 0.2,   # if moving mask covers <= this fraction, treat as not confidently moving
        hot_overlap_min: float = 0.5,       # and if hot-mask covers >= this fraction, re-label as static
    ) -> None:
        self.name = "blob"
        self.buffer_size = max(3, buffer_size)
        self.min_area = max(1, min_area)
        self.kernel_open = kernel_open
        self.kernel_close = kernel_close
        self.dilate_iter = dilate_iter
        self.draw = draw
        self.label = label
        self._buf_c = deque(maxlen=self.buffer_size)  # Celsius background buffer
        self._buf_gray = deque(maxlen=self.buffer_size)  # Grayscale background buffer
        self.rel_mode = rel_mode
        self.delta_c = float(delta_c)
        # Derived grayscale thresholds (no explicit public args)
        self.delta_gray = int(max(1, round(self.delta_c * 6)))
        self.combine = combine
        self.motion_delta_c = float(motion_delta_c)
        self.motion_delta_gray = int(max(1, round(self.motion_delta_c * 6)))
        # spatial hot-spot
        self.hot_static = bool(hot_static)
        self.hot_kernel = int(max(3, hot_kernel) | 1)  # ensure odd and >=3
        self.hot_delta_c = float(hot_delta_c)
        self.hot_delta_gray = int(max(1, round(self.hot_delta_c * 6)))
        # area filtering (pixels)
        self.max_area = int(max(0, max_area))
        self.static_min_area = int(max(0, static_min_area))
        self.static_min_area_mult = float(max(1.0, static_min_area_mult))
        nk = int(noise_kernel)
        self.noise_kernel = 1 if nk <= 1 else (nk | 1)  # ensure odd size, 1 disables
        self.bg_freeze_on_detect = bool(bg_freeze_on_detect)
        # tracking/persistence state
        self.persist_min_frames = max(1, int(persist_min_frames))
        self.persist_min_seconds = float(max(0.0, persist_min_seconds))
        self.track_max_miss = max(0, int(track_max_miss))
        self.track_iou_thresh = float(max(0.0, min(1.0, track_iou_thresh)))
        self.static_after_frames = max(0, int(static_after_frames))
        self.moving_after_frames = max(0, int(moving_after_frames))
        self.static_pos_px = int(max(0, static_pos_px))
        self.static_area_frac = float(max(0.0, min(1.0, static_area_frac)))
        self.static_iou = float(max(0.0, min(1.0, static_iou)))
        self.moving_coverage_max = float(max(0.0, min(1.0, moving_coverage_max)))
        self.hot_overlap_min = float(max(0.0, min(1.0, hot_overlap_min)))
        self._tracks = []  # list of dicts: {id, bbox, hits, misses, first_ts, last_ts}
        self._next_id = 1
        self._warned_no_celsius = False

    def process(self, frame: Frame, ctx: Dict[str, object]) -> None:
        # Acquire domains
        has_c = frame.celsius is not None
        has_gray = frame.gray is not None
        curr_c = frame.celsius.astype(np.float32, copy=False) if has_c else None
        curr_gray = frame.gray if has_gray else (cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY) if frame.bgr is not None else None)

        # Apply noise suppression
        if has_c:
            curr_c_proc = cv2.GaussianBlur(curr_c, (self.noise_kernel, self.noise_kernel), 0) if self.noise_kernel > 1 else curr_c
        else:
            curr_c_proc = None
        if curr_gray is not None:
            curr_gray_proc = cv2.GaussianBlur(curr_gray, (self.noise_kernel, self.noise_kernel), 0) if self.noise_kernel > 1 else curr_gray
        else:
            curr_gray_proc = None

        # Background readiness (keep indent inside method)
        has_bg_c = has_c and (len(self._buf_c) == self._buf_c.maxlen)
        # If Celsius is available, do NOT use grayscale for detection (thermal-only policy)
        use_gray = (not has_c)
        has_bg_gray = use_gray and (curr_gray is not None) and (len(self._buf_gray) == self._buf_gray.maxlen)

        bg_c = bg_c_proc = diff_c = None
        if has_bg_c:
            bg_c = np.median(np.stack(self._buf_c, axis=0), axis=0).astype(np.float32)
            bg_c_proc = cv2.GaussianBlur(bg_c, (self.noise_kernel, self.noise_kernel), 0) if self.noise_kernel > 1 else bg_c
            diff_c = np.abs(curr_c_proc - bg_c_proc)

        bg_gray = bg_gray_proc = diff_gray = None
        if has_bg_gray:
            bg_gray = np.median(np.stack(self._buf_gray, axis=0), axis=0).astype(np.uint8)
            bg_gray_proc = cv2.GaussianBlur(bg_gray, (self.noise_kernel, self.noise_kernel), 0) if self.noise_kernel > 1 else bg_gray
            diff_gray = cv2.absdiff(curr_gray_proc, bg_gray_proc)

        # Motion mask preference: Celsius then grayscale fallback
        if diff_c is not None:
            motion_mask = (diff_c >= self.motion_delta_c).astype(np.uint8) * 255
        elif use_gray and diff_gray is not None:
            motion_mask = (diff_gray >= self.motion_delta_gray).astype(np.uint8) * 255
        else:
            # choose a safe shape: prefer grayscale, then celsius, then bgr, else 1x1
            if use_gray and (curr_gray_proc is not None):
                base_shape = curr_gray_proc
            elif curr_c_proc is not None:
                base_shape = curr_c_proc
            elif frame.bgr is not None:
                base_shape = frame.bgr[..., 0]
            else:
                base_shape = np.zeros((1, 1), dtype=np.uint8)
            motion_mask = np.zeros_like(base_shape, dtype=np.uint8)

        # Relative temporal mask
        rel_mask = None
        if self.rel_mode != "off":
            if diff_c is not None and bg_c_proc is not None:
                delta = curr_c_proc - bg_c_proc
                if self.rel_mode == "hot":
                    rel_mask = (delta >= self.delta_c).astype(np.uint8) * 255
                elif self.rel_mode == "cold":
                    rel_mask = (delta <= -self.delta_c).astype(np.uint8) * 255
                else:
                    rel_mask = (np.abs(delta) >= self.delta_c).astype(np.uint8) * 255
            elif use_gray and (diff_gray is not None):
                # grayscale only supports magnitude comparison
                rel_mask = (diff_gray >= self.delta_gray).astype(np.uint8) * 255

        # Spatial hot-spot (warm static) mask
        hot_mask = None
        if self.hot_static:
            if curr_c_proc is not None:
                blur_c = cv2.GaussianBlur(curr_c_proc, (self.hot_kernel, self.hot_kernel), 0)
                hot_mask = (curr_c_proc - blur_c >= self.hot_delta_c).astype(np.uint8) * 255
            elif use_gray and (curr_gray_proc is not None):
                blur_g = cv2.GaussianBlur(curr_gray_proc, (self.hot_kernel, self.hot_kernel), 0)
                diff_hot = cv2.subtract(curr_gray_proc, blur_g)
                hot_mask = (diff_hot >= self.hot_delta_gray).astype(np.uint8) * 255

        mask = motion_mask
        if rel_mask is not None:
            if self.combine == "intersect":
                mask = cv2.bitwise_and(motion_mask, rel_mask)
            else:
                mask = cv2.bitwise_or(motion_mask, rel_mask)

        moving_mask = mask
        static_mask = None
        if hot_mask is not None:
            inv_moving = cv2.bitwise_not(moving_mask)
            static_mask = cv2.bitwise_and(hot_mask, inv_moving)

        def _morph(m: np.ndarray) -> np.ndarray:
            mm = m
            if self.kernel_open > 1:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_open, self.kernel_open))
                mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, k)
            if self.kernel_close > 1:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_close, self.kernel_close))
                mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, k)
            if self.dilate_iter > 0:
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                mm = cv2.dilate(mm, k, iterations=self.dilate_iter)
            return mm

        moving_mask = _morph(moving_mask)
        if static_mask is not None:
            static_mask = _morph(static_mask)

        # Contours per category
        det_boxes_moving: List[Tuple[int, int, int, int]] = []
        det_boxes_static: List[Tuple[int, int, int, int]] = []
        contours_m, _ = cv2.findContours(moving_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area_px = self.max_area if (self.max_area and self.max_area > 0) else None
        min_thresh_m = int(self.min_area)
        # static min area: explicit override or multiplier
        min_thresh_s = int(self.static_min_area if self.static_min_area > 0 else self.min_area * self.static_min_area_mult)

        for c in contours_m:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < min_thresh_m:
                continue
            if max_area_px is not None and area > max_area_px:
                continue
            det_boxes_moving.append((x, y, x + w, y + h))

        if static_mask is not None:
            contours_s, _ = cv2.findContours(static_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours_s:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                if area < min_thresh_s:
                    continue
                if max_area_px is not None and area > max_area_px:
                    continue
                det_boxes_static.append((x, y, x + w, y + h))

        # Suppress moving boxes that are fully contained in larger moving boxes
        if det_boxes_moving:
            def _area(b):
                return max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))

            def _contains(outer, inner):
                return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

            det_boxes_moving_sorted = sorted(det_boxes_moving, key=_area, reverse=True)
            kept_moving: List[Tuple[int, int, int, int]] = []
            for b in det_boxes_moving_sorted:
                discard = False
                for kb in kept_moving:
                    if _contains(kb, b):
                        discard = True
                        break
                if not discard:
                    kept_moving.append(b)
            det_boxes_moving = kept_moving

        # Optionally re-label weak moving blobs (low motion coverage) as static if hot evidence is strong
        if det_boxes_moving and hot_mask is not None and self.moving_coverage_max > 0.0:
            relabeled_static: List[Tuple[int, int, int, int]] = []
            kept_moving2: List[Tuple[int, int, int, int]] = []
            for (x1, y1, x2, y2) in det_boxes_moving:
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                box_area = float(w * h)
                mv_roi = moving_mask[y1:y2, x1:x2]
                ht_roi = hot_mask[y1:y2, x1:x2]
                mv_ratio = float(np.count_nonzero(mv_roi)) / box_area
                ht_ratio = float(np.count_nonzero(ht_roi)) / box_area
                if mv_ratio <= self.moving_coverage_max and ht_ratio >= self.hot_overlap_min:
                    # meets static evidence; only keep if passes static min area threshold
                    if (w * h) >= min_thresh_s and (max_area_px is None or (w * h) <= max_area_px):
                        relabeled_static.append((x1, y1, x2, y2))
                else:
                    kept_moving2.append((x1, y1, x2, y2))
            # merge relabeled into static list
            if relabeled_static:
                det_boxes_static.extend(relabeled_static)
            det_boxes_moving = kept_moving2

        # Discard static boxes fully contained within any larger box (moving or static)
        if det_boxes_static:
            def _area(b):
                return max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))

            def _contains(outer, inner):
                return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

            all_boxes = det_boxes_moving + det_boxes_static
            kept_static: List[Tuple[int, int, int, int]] = []
            for s in det_boxes_static:
                s_area = _area(s)
                discard = False
                for b in all_boxes:
                    if b is s:
                        continue
                    if _area(b) > s_area and _contains(b, s):
                        discard = True
                        break
                if not discard:
                    kept_static.append(s)
            det_boxes_static = kept_static

        # Combine detections with labels for tracking
        dets_labeled: List[Tuple[int, int, int, int, str]] = []
        for b in det_boxes_moving:
            dets_labeled.append((b[0], b[1], b[2], b[3], 'moving'))
        for b in det_boxes_static:
            dets_labeled.append((b[0], b[1], b[2], b[3], 'static'))

        # Update tracks and filter by persistence (preserve label)
        persisted_labeled = self._update_tracks_and_get_persisted(dets_labeled, frame.timestamp)

        # If YOLO detections are available, drop moving blobs fully contained inside any YOLO box
        try:
            yolo_boxes_raw = ctx.get('yolo', []) if isinstance(ctx, dict) else []
            yolo_boxes = []
            for d in yolo_boxes_raw:
                if isinstance(d, (list, tuple)) and len(d) >= 4:
                    yolo_boxes.append((int(d[0]), int(d[1]), int(d[2]), int(d[3])))
            def _contained(outer: Tuple[int,int,int,int], inner: Tuple[int,int,int,int]) -> bool:
                ox1, oy1, ox2, oy2 = outer
                ix1, iy1, ix2, iy2 = inner
                return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2
            if yolo_boxes:
                filtered = []
                for (x1, y1, x2, y2, typ) in persisted_labeled:
                    if typ == 'moving':
                        in_any = any(_contained(yb, (x1, y1, x2, y2)) for yb in yolo_boxes)
                        if in_any:
                            continue  # suppress moving blob inside YOLO
                    filtered.append((x1, y1, x2, y2, typ))
                persisted_labeled = filtered
        except Exception:
            # if anything goes wrong with ctx/yolo format, keep original list
            pass

        ctx[self.name] = persisted_labeled

        if self.draw:
            for (x1, y1, x2, y2, typ) in persisted_labeled:
                if typ == 'moving':
                    color = (255, 255, 255)  # white for moving as requested
                else:
                    color = (255, 255, 0)  # cyan/yellow for static
                cv2.rectangle(frame.bgr, (x1, y1), (x2, y2), color, 1)
                org = (x1, max(12, y1 - 4))
                # outline text for readability
                cv2.putText(frame.bgr, typ, org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2, cv2.LINE_8)
                cv2.putText(frame.bgr, typ, org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_8)

        # Update buffers AFTER processing, optionally freeze on detection to avoid absorbing static objects
        if not (self.bg_freeze_on_detect and len(persisted_labeled) > 0):
            if curr_c is not None:
                self._buf_c.append(curr_c.copy())
            if use_gray and (curr_gray is not None):
                self._buf_gray.append(curr_gray.copy())

    # --- Tracking helpers ---
    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter) / float(union) if union > 0 else 0.0

    def _update_tracks_and_get_persisted(self, dets: List[Tuple[int, int, int, int, str]], ts: float) -> List[Tuple[int, int, int, int, str]]:
        # Match detections to existing tracks by IoU
        unmatched_dets = set(range(len(dets)))
        # Prepare track indices
        for t in self._tracks:
            t['matched'] = False

        # Greedy matching
        for ti, t in enumerate(self._tracks):
            best_j = -1
            best_iou = 0.0
            for j in list(unmatched_dets):
                iou = self._iou(t['bbox'], dets[j][:4])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= self.track_iou_thresh:
                # Update track
                prev_bbox = t['bbox']
                new_bbox = dets[best_j][:4]
                t['hits'] += 1
                t['misses'] = 0
                t['last_ts'] = ts
                # Update current label and static streak
                cur_label = dets[best_j][4]
                t['label_current'] = cur_label
                # Geometric stability check (center shift, area change, IoU)
                stable_geom = self._bbox_stable(prev_bbox, new_bbox)
                # Evidence accumulation: treat geometric stability as static evidence
                static_evidence = (cur_label == 'static') or stable_geom
                moving_evidence = (cur_label == 'moving') and not stable_geom
                # Update streaks with decay to resist flicker
                if static_evidence:
                    t['static_streak'] = t.get('static_streak', 0) + 1
                    t['moving_streak'] = max(0, t.get('moving_streak', 0) - 1)
                elif moving_evidence:
                    t['moving_streak'] = t.get('moving_streak', 0) + 1
                    t['static_streak'] = max(0, t.get('static_streak', 0) - 1)
                else:
                    # neutral: mild decay on both
                    t['static_streak'] = max(0, t.get('static_streak', 0) - 1)
                    t['moving_streak'] = max(0, t.get('moving_streak', 0) - 1)

                # Effective label with hysteresis
                prev_eff = t.get('effective', 'moving')
                eff = prev_eff
                if eff == 'moving':
                    if self.static_after_frames == 0 or t['static_streak'] >= self.static_after_frames:
                        eff = 'static'
                else:  # currently static
                    if self.moving_after_frames == 0 or t['moving_streak'] >= self.moving_after_frames:
                        eff = 'moving'
                t['effective'] = eff
                # Reset opposite evidence streak on transition to avoid immediate flip-back oscillation.
                if prev_eff != eff:
                    if eff == 'moving':
                        t['static_streak'] = 0
                    elif eff == 'static':
                        t['moving_streak'] = 0
                # If just converted to static but below static min area, mark for pruning
                if prev_eff != 'static' and eff == 'static':
                    pw = max(1, new_bbox[2] - new_bbox[0])
                    ph = max(1, new_bbox[3] - new_bbox[1])
                    area = pw * ph
                    static_min_px = int(self.static_min_area if self.static_min_area > 0 else self.min_area * self.static_min_area_mult)
                    if area < static_min_px:
                        t['prune'] = True
                # Finally update bbox after using prev for stability calc
                t['bbox'] = new_bbox
                t['matched'] = True
                unmatched_dets.discard(best_j)
            else:
                # No match for this track this frame
                t['misses'] += 1

        # Create new tracks for unmatched detections
        for j in unmatched_dets:
            self._tracks.append({
                'id': self._next_id,
                'bbox': dets[j][:4],
                'label_current': dets[j][4],
                'static_streak': (1 if dets[j][4] == 'static' else 0),
                'moving_streak': (1 if dets[j][4] == 'moving' else 0),
                'effective': ('static' if (dets[j][4] == 'static' and self.static_after_frames == 0) else 'moving'),
                'hits': 1,
                'misses': 0,
                'first_ts': ts,
                'last_ts': ts,
            })
            self._next_id += 1

        # Prune stale tracks and those marked for pruning
        self._tracks = [t for t in self._tracks if (t.get('misses', 0) <= self.track_max_miss and not t.get('prune', False))]

        # Persist filter
        out: List[Tuple[int, int, int, int, str]] = []
        for t in self._tracks:
            age_ok = t['hits'] >= self.persist_min_frames
            time_ok = (self.persist_min_seconds > 0.0 and (t['last_ts'] - t['first_ts']) >= self.persist_min_seconds)
            if age_ok or time_ok:
                eff = t.get('effective', t.get('label_current', 'moving'))
                if eff == 'static':
                    bx1, by1, bx2, by2 = t['bbox']
                    area = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
                    static_min_px = int(self.static_min_area if self.static_min_area > 0 else self.min_area * self.static_min_area_mult)
                    if area < static_min_px:
                        continue
                out.append((*t['bbox'], eff))
        return out

    def _bbox_stable(self, prev: Tuple[int, int, int, int], cur: Tuple[int, int, int, int]) -> bool:
        """Return True if bbox movement/shape change is within static thresholds.
        Criteria:
          - IoU(prev, cur) >= self.static_iou, OR
          - center shift <= self.static_pos_px AND relative area change <= self.static_area_frac
        """
        # IoU criterion
        if self._iou(prev, cur) >= self.static_iou:
            return True

        px1, py1, px2, py2 = prev
        cx1, cy1, cx2, cy2 = cur
        pw, ph = max(1, px2 - px1), max(1, py2 - py1)
        cw, ch = max(1, cx2 - cx1), max(1, cy2 - cy1)
        pcx, pcy = px1 + pw * 0.5, py1 + ph * 0.5
        ccx, ccy = cx1 + cw * 0.5, cy1 + ch * 0.5
        shift = ((ccx - pcx) ** 2 + (ccy - pcy) ** 2) ** 0.5
        if shift > self.static_pos_px:
            return False
        p_area = pw * ph
        c_area = cw * ch
        if p_area <= 0:
            return False
        area_change = abs(c_area - p_area) / float(p_area)
        return area_change <= self.static_area_frac
