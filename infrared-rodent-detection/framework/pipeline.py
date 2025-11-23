from __future__ import annotations

import json
import time
import os
import signal
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol

import cv2
import numpy as np


@dataclass
class Frame:
    """A container for a single frame flowing through the pipeline."""
    bgr: np.ndarray  # primary visualization buffer (H x W x 3, uint8)
    gray: Optional[np.ndarray] = None  # optional grayscale view (H x W, uint8)
    celsius: Optional[np.ndarray] = None  # optional thermal data in Celsius (float32/float64)
    raw: Optional[np.ndarray] = None  # optional raw sensor data (e.g., uint16 thermal)
    timestamp: float = field(default_factory=lambda: time.time())
    meta: Dict[str, object] = field(default_factory=dict)


class _RawRecorder:
    """Writes raw uint16 frames sequentially to disk with optional metadata."""

    def __init__(self, path: str, meta_path: Optional[str] = None) -> None:
        self.path = path
        self.meta_path = meta_path or f"{path}.json"
        dir_path = os.path.dirname(os.path.abspath(path))
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        if self.meta_path:
            meta_dir = os.path.dirname(os.path.abspath(self.meta_path))
            if meta_dir and not os.path.exists(meta_dir):
                os.makedirs(meta_dir, exist_ok=True)
        self._fh = open(path, "wb")
        self.frame_count = 0
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.source_dtype: Optional[str] = None
        self.dtype: Optional[str] = None
        self.bytes_per_frame: Optional[int] = None

    def write(self, data: np.ndarray) -> None:
        if data.ndim != 2:
            raise ValueError("Raw recorder expects 2D frames")
        contiguous = np.ascontiguousarray(data)
        src_dtype = np.dtype(contiguous.dtype).str
        if contiguous.dtype != np.uint16:
            contiguous = contiguous.astype(np.uint16, copy=False)
        h, w = contiguous.shape
        frame_bytes = contiguous.nbytes
        if self.frame_count == 0:
            self.height = int(h)
            self.width = int(w)
            self.source_dtype = src_dtype
            self.dtype = np.dtype(contiguous.dtype).str
            self.bytes_per_frame = frame_bytes
        else:
            if self.height != h or self.width != w:
                raise ValueError("Raw recorder received frame with mismatched dimensions")
            if self.bytes_per_frame is not None and self.bytes_per_frame != frame_bytes:
                raise ValueError("Raw recorder received frame with mismatched byte size")
        self._fh.write(contiguous.tobytes())
        self.frame_count += 1

    def close(self) -> None:
        try:
            if self._fh:
                self._fh.close()
        finally:
            self._fh = None
        if self.meta_path:
            meta = {
                "path": os.path.abspath(self.path),
                "width": self.width,
                "height": self.height,
                "frame_count": self.frame_count,
                "written_dtype": self.dtype,
                "source_dtype": self.source_dtype,
                "bytes_per_frame": self.bytes_per_frame,
                "endianness": "little" if np.little_endian else "big",
                "notes": "Frames stored sequentially, row-major, no header."
            }
            with open(self.meta_path, "w", encoding="utf-8") as meta_f:
                json.dump(meta, meta_f, indent=2)


class FrameSource(Protocol):
    """Abstract interface for frame sources."""

    def start(self) -> None:
        ...

    def frames(self) -> Iterable[Frame]:
        ...

    def stop(self) -> None:
        ...


class Plugin(Protocol):
    """Plugins receive frames and may annotate them, publish data, or modify the frame in-place."""

    name: str

    def process(self, frame: Frame, ctx: Dict[str, object]) -> None:
        """Process a frame. May modify frame.bgr in-place for overlays or store results in ctx[name]."""
        ...


class Pipeline:
    """A simple, pluggable image processing pipeline.

    Contract:
    - Input: frames produced by a FrameSource.
    - Plugins: sequentially called per frame with shared ctx dict.
    - Output: optionally displayed window; users can also consume ctx.
    """

    def __init__(
        self,
        source: FrameSource,
        plugins: List[Plugin],
        *,
        window_name: str = "Pipeline",
        display: bool = True,
        show_fps: bool = True,
        esc_to_quit: bool = True,
        # Output recording
        save_path: Optional[str] = None,
        save_fps: Optional[float] = None,
        raw_path: Optional[str] = None,
        raw_meta_path: Optional[str] = None,
        # Object count overlay/options
        show_count: bool = True,
        count_reset_frames: int = 0,
        count_reset_sec: float = 0.0,
        max_playback_fps: Optional[float] = None,
        show_temp_overlay: bool = True,
        enable_overlays: bool = True,
    ) -> None:
        self.source = source
        self.plugins = plugins
        self.window_name = window_name
        self.display = display
        self.show_fps = show_fps
        self.esc_to_quit = esc_to_quit
        self._running = False
        # Recording
        self.save_path = save_path
        self.save_fps = save_fps
        self._writer = None
        self.raw_path = raw_path
        self.raw_meta_path = raw_meta_path
        self._raw_recorder: Optional[_RawRecorder] = None
        self._raw_recorder_failed = False
        self._raw_recorder_warned = False
        # Count overlay state
        self.show_count = show_count
        self.count_reset_frames = max(0, int(count_reset_frames))
        self.count_reset_sec = float(max(0.0, count_reset_sec))
        self._displayed_count = 0
        self._zero_streak = 0
        self._last_nonzero_time = 0.0
        self.max_playback_fps = float(max_playback_fps) if (max_playback_fps and max_playback_fps > 0) else None
        self.show_temp_overlay = bool(show_temp_overlay)
        self.enable_overlays = bool(enable_overlays)

    def run(self) -> None:
        self._running = True

        # Install a minimal Ctrl+C handler to request a graceful stop
        def _handle_sigint(signum, frame):  # noqa: ARG001
            self._running = False
        old_sigint = None
        try:
            old_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _handle_sigint)
        except Exception:
            # On some environments, setting signal handlers may not be supported
            pass

        # Ensure we do not overwrite existing output files; create unique paths
        try:
            if self.save_path:
                self.save_path = self._unique_output_path(self.save_path)
            if self.raw_path:
                self.raw_path = self._unique_output_path(self.raw_path)
        except Exception:
            # If uniqueness check fails for any reason, proceed with original paths
            pass

        self.source.start()

        if self.display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        last_t = time.time()
        fps = 0.0
        alpha = 0.9  # for EMA FPS
        ctx: Dict[str, object] = {}

        # Create an explicit iterator so we can close the generator on exit,
        # ensuring any internal "finally" blocks (e.g., releasing camera images) run.
        frames_iter = self.source.frames()
        try:
            for frame in frames_iter:
                frame_loop_start = time.time()
                if not self._running:
                    break

                # Update FPS
                now = frame_loop_start
                dt = max(1e-6, now - last_t)
                last_t = now
                fps = alpha * fps + (1 - alpha) * (1.0 / dt)

                # Persist raw thermal frames before plugins modify the visualization
                if self.raw_path and not self._raw_recorder_failed:
                    raw_frame = frame.raw
                    if raw_frame is None:
                        if not self._raw_recorder_warned:
                            ctx.setdefault("warn:raw_recorder", "Raw data unavailable in frames; skipping raw recording")
                            self._raw_recorder_warned = True
                    else:
                        if self._raw_recorder is None:
                            try:
                                meta_path = self.raw_meta_path if self.raw_meta_path else None
                                self._raw_recorder = _RawRecorder(self.raw_path, meta_path)
                            except Exception as ex:
                                ctx.setdefault("error:raw_recorder", str(ex))
                                self._raw_recorder_failed = True
                        if self._raw_recorder is not None:
                            try:
                                self._raw_recorder.write(raw_frame)
                            except Exception as ex:
                                ctx.setdefault("error:raw_recorder", str(ex))
                                self._raw_recorder_failed = True
                                try:
                                    self._raw_recorder.close()
                                except Exception:
                                    pass
                                self._raw_recorder = None

                # Process plugins sequentially
                for plugin in self.plugins:
                    try:
                        plugin.process(frame, ctx)
                    except Exception as ex:
                        # Keep pipeline resilient; record the error
                        ctx[f"error:{plugin.name}"] = str(ex)

                # Compute object counts from plugins with de-duplication and static filtering
                measured_count = 0
                try:
                    yolo_boxes = []
                    blob_moving = []
                    if 'yolo' in ctx and isinstance(ctx['yolo'], list):
                        # YOLO entries: (x1,y1,x2,y2,conf,cls)
                        for d in ctx['yolo']:
                            if isinstance(d, (list, tuple)) and len(d) >= 4:
                                yolo_boxes.append((int(d[0]), int(d[1]), int(d[2]), int(d[3])))
                    if 'blob' in ctx and isinstance(ctx['blob'], list):
                        # Blob entries: (x1,y1,x2,y2,label)
                        for d in ctx['blob']:
                            if isinstance(d, (list, tuple)) and len(d) >= 5:
                                if str(d[4]).lower() != 'static':
                                    blob_moving.append((int(d[0]), int(d[1]), int(d[2]), int(d[3])))

                    if yolo_boxes and blob_moving:
                        measured_count = self._merged_count(yolo_boxes, blob_moving, iou_thresh=0.5)
                    elif yolo_boxes:
                        measured_count = len(yolo_boxes)
                    else:
                        measured_count = len(blob_moving)
                except Exception:
                    measured_count = 0

                now_ts = time.time()
                if measured_count > 0:
                    self._displayed_count = int(measured_count)
                    self._zero_streak = 0
                    self._last_nonzero_time = now_ts
                else:
                    self._zero_streak += 1
                    # keep previous displayed count until threshold is reached
                    if (self.count_reset_frames and self._zero_streak >= self.count_reset_frames) or \
                       (self.count_reset_sec and (now_ts - self._last_nonzero_time) >= self.count_reset_sec):
                        self._displayed_count = 0

                # Compute min/max Celsius if available
                temp_text = None
                if self.enable_overlays and self.show_temp_overlay and frame.celsius is not None:
                    try:
                        cmin = float(np.min(frame.celsius))
                        cmax = float(np.max(frame.celsius))
                        temp_text = f"Temp C: min={cmin:5.2f} max={cmax:5.2f}"
                    except Exception:
                        temp_text = None

                # Render
                if self.display:
                    vis = frame.bgr
                    if self.enable_overlays:
                        if self.show_fps:
                            cv2.putText(
                                vis,
                                f"FPS: {fps:5.1f}",
                                (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (0, 255, 0),
                                1,
                                cv2.LINE_8,
                            )
                        if self.show_count:
                            cv2.putText(
                                vis,
                                f"Objects: {self._displayed_count}",
                                (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (0, 200, 255),
                                1,
                                cv2.LINE_8,
                            )
                        if temp_text:
                            cv2.putText(
                                vis,
                                temp_text,
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (255, 255, 0),
                                1,
                                cv2.LINE_8,
                            )
                    # Initialize writer lazily when we know frame size
                    if self.save_path and self._writer is None:
                        try:
                            h, w = vis.shape[:2]
                            # Use fixed FPS unless explicitly overridden
                            out_fps = float(self.save_fps) if (self.save_fps and self.save_fps > 0) else 30.0
                            fourcc = self._choose_fourcc(self.save_path)
                            self._writer = cv2.VideoWriter(self.save_path, fourcc, out_fps, (w, h))
                        except Exception:
                            self._writer = None
                    # If user has closed the window (clicked the X), exit cleanly
                    try:
                        prop = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
                        if prop <= 0:  # window closed or not visible
                            self._running = False
                            break
                    except Exception:
                        # If the property cannot be queried, assume window is gone
                        self._running = False
                        break
                    # Show frame; if imshow fails (window closed), exit gracefully
                    try:
                        cv2.imshow(self.window_name, vis)
                    except Exception:
                        self._running = False
                        break
                    # Write frame to file if enabled
                    if self._writer is not None:
                        try:
                            self._writer.write(vis)
                        except Exception:
                            pass

                    # Process UI events
                    try:
                        key = cv2.waitKey(1) & 0xFF
                    except Exception:
                        self._running = False
                        break
                    # Allow both ESC and 'q' to quit
                    if self.esc_to_quit and (key == 27 or key == ord('q') or key == ord('Q')):
                        self._running = False
                        break
                    # Detect window close after event processing as well
                    try:
                        prop_after = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
                        if prop_after <= 0:
                            self._running = False
                            break
                    except Exception:
                        self._running = False
                        break

                # Rate limit playback if requested
                if self.max_playback_fps is not None:
                    target_dt = 1.0 / self.max_playback_fps
                    elapsed = time.time() - frame_loop_start
                    remaining = target_dt - elapsed
                    if remaining > 0:
                        try:
                            time.sleep(remaining)
                        except Exception:
                            pass

        finally:
            # Close generator explicitly to trigger inner cleanups
            try:
                close_fn = getattr(frames_iter, 'close', None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
            self._running = False
            self.source.stop()
            if self.display:
                try:
                    cv2.destroyWindow(self.window_name)
                    # In some OpenCV builds on Windows, destroying the named window may recreate
                    # a default window on next imshow call; ensure all are closed here.
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            # Release writer
            try:
                if self._writer is not None:
                    self._writer.release()
            except Exception:
                pass
            # Release raw recorder
            try:
                if self._raw_recorder is not None:
                    self._raw_recorder.close()
            except Exception:
                pass
            # Restore previous SIGINT handler if we set one
            try:
                if old_sigint is not None:
                    signal.signal(signal.SIGINT, old_sigint)
            except Exception:
                pass

    @staticmethod
    def _choose_fourcc(path: str) -> int:
        ext = os.path.splitext(path)[1].lower() if 'os' in globals() else ''
        try:
            import os as _os
            ext = _os.path.splitext(path)[1].lower()
        except Exception:
            pass
        if ext in ('.mp4', '.m4v', '.mov'):
            return cv2.VideoWriter_fourcc(*'mp4v')
        if ext in ('.avi',):
            return cv2.VideoWriter_fourcc(*'XVID')
        # default to mp4v
        return cv2.VideoWriter_fourcc(*'mp4v')

    @staticmethod
    def _unique_output_path(path: str) -> str:
        """Return a non-existing path by appending an incremental suffix before the extension.

        Example: results/output.mp4 -> results/output_001.mp4 if output.mp4 exists.
        """
        try:
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                return abs_path
            base, ext = os.path.splitext(abs_path)
            # Try numeric suffixes
            for i in range(1, 1000):
                candidate = f"{base}_{i:03d}{ext}"
                if not os.path.exists(candidate):
                    return candidate
            # Fallback to timestamp
            ts = time.strftime("%Y%m%d-%H%M%S")
            candidate = f"{base}_{ts}{ext}"
            return candidate
        except Exception:
            return path

    @staticmethod
    def _iou_xyxy(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = area_a + area_b - inter
        return float(inter) / float(denom) if denom > 0 else 0.0

    def _merged_count(self, a_boxes, b_boxes, iou_thresh: float = 0.5) -> int:
        # Count unique boxes from A union B, merging overlaps
        count = len(a_boxes)
        for bb in b_boxes:
            dup = False
            for aa in a_boxes:
                if self._iou_xyxy(aa, bb) >= iou_thresh:
                    dup = True
                    break
            if not dup:
                count += 1
        return count
