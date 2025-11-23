from __future__ import annotations

import os
from typing import Iterable, Optional, Union

import cv2
import numpy as np

from ..pipeline import Frame, FrameSource


class OpenCVSource(FrameSource):
    """FrameSource using cv2.VideoCapture for camera/video or a folder of images.

    Args:
        input: int for camera index, str path to video file, image file, or folder of images.
        loop: if True, loops when reaching the end (for files/folders).
        resize: optional (width, height) to resize frames for downstream.
    """

    def __init__(
        self,
        input: Union[int, str] = 0,
        *,
        loop: bool = True,
        resize: Optional[tuple[int, int]] = None,
    ) -> None:
        self.input = input
        self.loop = loop
        self.resize = resize
        self._cap: Optional[cv2.VideoCapture] = None
        self._image_files: Optional[list[str]] = None
        self._running = False

    def start(self) -> None:
        if isinstance(self.input, int) or (isinstance(self.input, str) and self._is_video_file(self.input)):
            # Camera index or video file
            self._cap = cv2.VideoCapture(self.input)
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open VideoCapture for {self.input}")
        elif isinstance(self.input, str):
            if os.path.isdir(self.input):
                exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
                files = [os.path.join(self.input, f) for f in sorted(os.listdir(self.input)) if os.path.splitext(f)[1].lower() in exts]
                if not files:
                    raise RuntimeError(f"No images found in folder: {self.input}")
                self._image_files = files
            elif os.path.isfile(self.input):
                self._image_files = [self.input]
            else:
                raise RuntimeError(f"Invalid input path: {self.input}")
        else:
            raise RuntimeError("Unsupported input type for OpenCVSource")
        self._running = True

    def frames(self) -> Iterable[Frame]:
        if not self._running:
            return

        import time

        if self._cap is not None:
            # Camera/video
            while self._running:
                ok, frame = self._cap.read()
                if not ok:
                    if self.loop and isinstance(self.input, str):
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                if self.resize is not None:
                    frame = cv2.resize(frame, self.resize)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield Frame(bgr=frame, gray=gray)
        else:
            # Image files
            assert self._image_files is not None
            idx = 0
            n = len(self._image_files)
            while self._running:
                path = self._image_files[idx]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    # Skip bad files
                    idx = (idx + 1) % n
                    continue
                if self.resize is not None:
                    img = cv2.resize(img, self.resize)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                yield Frame(bgr=img, gray=gray, meta={"path": path})
                idx += 1
                if idx >= n:
                    if self.loop:
                        idx = 0
                    else:
                        break

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

    @staticmethod
    def _is_video_file(path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
