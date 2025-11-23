from __future__ import annotations

import json
import os
import time
from typing import Iterable, Optional

import cv2
import numpy as np

from ..pipeline import Frame, FrameSource


class RawFileSource(FrameSource):
    """FrameSource that replays raw uint16 thermal frames from disk."""

    def __init__(
        self,
        raw_path: Optional[str],
        *,
        meta_path: Optional[str] = None,
        loop: bool = False,
        override_fps: Optional[float] = None,
    ) -> None:
        if not raw_path and not meta_path:
            raise ValueError("RawFileSource requires at least a raw path or metadata path")
        self.raw_path = raw_path
        self.meta_path = meta_path
        self.loop = loop
        self.override_fps = override_fps if (override_fps and override_fps > 0) else None

        self._running = False
        self._memmap: Optional[np.memmap] = None
        self._dtype: Optional[np.dtype] = None
        self._frame_pixels: Optional[int] = None
        self._frame_count: Optional[int] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._fps: Optional[float] = None
        self._meta: dict[str, object] | None = None
        self._start_time: float = 0.0

    def start(self) -> None:
        raw_path: Optional[str] = self.raw_path
        meta_path = self.meta_path
        if meta_path is None and raw_path:
            guess = f"{raw_path}.json"
            if os.path.isfile(guess):
                meta_path = guess
        meta: dict[str, object] = {}
        if meta_path:
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(f"Raw metadata JSON not found: {meta_path}")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        if raw_path is None:
            candidate = meta.get("path") if isinstance(meta, dict) else None
            if not candidate or not isinstance(candidate, str):
                raise ValueError("Metadata must include 'path' to raw data when raw path not provided")
            raw_path = candidate
        if not os.path.isfile(raw_path):
            raise FileNotFoundError(f"Raw file not found: {raw_path}")

        width = int(meta.get("width", 0)) if isinstance(meta, dict) else 0
        height = int(meta.get("height", 0)) if isinstance(meta, dict) else 0
        if width <= 0 or height <= 0:
            raise ValueError("Metadata must include positive 'width' and 'height'")
        dtype_str = "<u2"
        if isinstance(meta, dict):
            dtype_str = str(meta.get("written_dtype") or meta.get("dtype") or dtype_str)
        dtype = np.dtype(dtype_str)
        frame_pixels = width * height
        if frame_pixels <= 0:
            raise ValueError("Invalid frame dimensions in metadata")

        file_size = os.path.getsize(raw_path)
        frame_bytes = frame_pixels * dtype.itemsize
        if frame_bytes <= 0:
            raise ValueError("Invalid dtype or frame size for raw data")
        if file_size % frame_bytes != 0:
            raise ValueError("Raw file size is not an integer multiple of frame size; metadata mismatch")
        frame_count = file_size // frame_bytes
        meta_count = int(meta.get("frame_count", 0)) if isinstance(meta, dict) else 0
        if meta_count and meta_count != frame_count:
            frame_count = min(frame_count, meta_count)

        fps_val = None
        if isinstance(meta, dict) and "fps" in meta:
            try:
                fps_val = float(meta["fps"])
            except Exception:
                fps_val = None
        if self.override_fps is not None:
            fps_val = self.override_fps
        elif fps_val is not None and fps_val <= 0:
            fps_val = None

        self._memmap = np.memmap(raw_path, mode="r", dtype=dtype)
        self._dtype = dtype
        self._frame_pixels = frame_pixels
        self._frame_count = int(frame_count)
        self._width = width
        self._height = height
        self._fps = fps_val
        self._meta = meta
        self._running = True
        self._start_time = time.time()

    def frames(self) -> Iterable[Frame]:
        if not self._running:
            return
        assert self._memmap is not None
        assert self._dtype is not None
        assert self._frame_pixels is not None
        assert self._frame_count is not None
        assert self._width is not None
        assert self._height is not None

        idx = 0
        while self._running:
            if idx >= self._frame_count:
                if self.loop:
                    idx = 0
                else:
                    break
            start = idx * self._frame_pixels
            end = start + self._frame_pixels
            raw_view = self._memmap[start:end]
            frame16 = np.array(raw_view, dtype=self._dtype, copy=True).reshape(self._height, self._width)
            raw_frame = frame16.copy()
            celsius = raw_frame.astype(np.float32) * 0.04 - 273.15

            cmin = float(np.min(celsius))
            cmax = float(np.max(celsius))
            if cmax - cmin <= 1e-6:
                gray8 = np.zeros_like(raw_frame, dtype=np.uint8)
            else:
                gray8 = np.clip(((celsius - cmin) / (cmax - cmin) * 255.0), 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)

            if self._fps:
                timestamp = self._start_time + idx / self._fps
            else:
                timestamp = time.time()

            meta = {"frame_index": idx}
            yield Frame(bgr=bgr, gray=gray8, celsius=celsius, raw=raw_frame, timestamp=timestamp, meta=meta)
            idx += 1

    def stop(self) -> None:
        self._running = False
        self._memmap = None
        self._dtype = None
        self._frame_pixels = None
        self._frame_count = None
        self._width = None
        self._height = None
        self._fps = None
        self._meta = None
