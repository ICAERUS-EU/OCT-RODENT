from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

try:
    import PySpin  # type: ignore
except Exception as _ex:  # pragma: no cover
    PySpin = None  # type: ignore

from ..pipeline import Frame, FrameSource


def _set_thermal_properties(nodemap) -> bool:
    """Mirror of realtime_yolo.set_thermal_properties, localized to avoid import cycles."""
    try:
        node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
        node_pixel_format_mono14 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('Mono14'))
        node_pixel_format.SetIntValue(node_pixel_format_mono14.GetValue())

        node_temp_linear = PySpin.CEnumerationPtr(nodemap.GetNode('TemperatureLinearResolution'))
        node_temp_linear_high = PySpin.CEnumEntryPtr(node_temp_linear.GetEntryByName('High'))
        node_temp_linear.SetIntValue(node_temp_linear_high.GetValue())

        node_bit_depth = PySpin.CEnumerationPtr(nodemap.GetNode('CMOSBitDepth'))
        node_bit_depth_14bit = PySpin.CEnumEntryPtr(node_bit_depth.GetEntryByName('bit14bit'))
        node_bit_depth.SetIntValue(node_bit_depth_14bit.GetValue())

        node_temp_linear_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TemperatureLinearMode'))
        node_temp_linear_on = PySpin.CEnumEntryPtr(node_temp_linear_mode.GetEntryByName('On'))
        node_temp_linear_mode.SetIntValue(node_temp_linear_on.GetValue())
        return True
    except Exception:
        return False


class SpinnakerSource(FrameSource):
    """FrameSource that streams frames from a FLIR AX5 via PySpin.

    It yields frames with:
        - celsius: float32 array computed from raw Mono14 data: C = raw*0.04 - 273.15
        - bgr: uint8 visualization derived from Celsius values (mapped 0..255 as 3-channel BGR)
    """

    def __init__(self, *, camera_index: int = 0, auto_configure: bool = True):
        if PySpin is None:
            raise RuntimeError("PySpin is not available; cannot create SpinnakerSource")
        self.camera_index = camera_index
        self.auto_configure = auto_configure

        self._system = None
        self._cam_list = None
        self._cam = None
        self._nodemap = None
        self._nodemap_tldevice = None
        self._running = False

    def start(self) -> None:
        self._system = PySpin.System.GetInstance()
        self._cam_list = self._system.GetCameras()
        try:
            if self._cam_list.GetSize() == 0:
                raise RuntimeError("No FLIR/Spinnaker cameras detected")
            # Note: camera_index is best-effort index into cam_list
            self._cam = self._cam_list[self.camera_index]
            self._cam.Init()
            self._nodemap = self._cam.GetNodeMap()
            self._nodemap_tldevice = self._cam.GetTLDeviceNodeMap()

            if self.auto_configure:
                _set_thermal_properties(self._nodemap)

            sNodemap = self._cam.GetTLStreamNodeMap()
            node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
            node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
            node_bufferhandling_mode.SetIntValue(node_newestonly.GetValue())

            node_acquisition_mode = PySpin.CEnumerationPtr(self._nodemap.GetNode('AcquisitionMode'))
            node_acq_cont = node_acquisition_mode.GetEntryByName('Continuous')
            node_acquisition_mode.SetIntValue(node_acq_cont.GetValue())

            self._cam.BeginAcquisition()
            self._running = True
        finally:
            # Do not clear cam_list until stop
            pass

    def frames(self) -> Iterable[Frame]:
        if not self._running:
            return
        import cv2

        while self._running:
            # Use a timeout to periodically check _running and exit cleanly
            try:
                image_result = self._cam.GetNextImage(200)  # timeout in ms
            except Exception:
                # If acquisition has ended or timed out, check running flag
                if not self._running:
                    break
                else:
                    continue
            try:
                if image_result.IsIncomplete():
                    continue
                raw16 = np.array(image_result.GetNDArray(), dtype=np.uint16, copy=True)
                # Celsius conversion: float32 for efficiency
                celsius = raw16.astype(np.float32) * 0.04 - 273.15
                # Normalize to 0..255 for visualization
                minv = float(np.min(celsius))
                maxv = float(np.max(celsius))
                if maxv - minv <= 1e-6:
                    gray8 = np.zeros_like(raw16, dtype=np.uint8)
                else:
                    gray8 = np.clip(((celsius - minv) / (maxv - minv) * 255.0), 0, 255).astype(np.uint8)
                bgr = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
                yield Frame(bgr=bgr, gray=gray8, celsius=celsius, raw=raw16, meta={})
            finally:
                image_result.Release()

    def stop(self) -> None:
        # Stop acquisition and clean up in safest order recommended by Spinnaker samples
        try:
            if self._cam is not None:
                try:
                    if self._running:
                        try:
                            self._cam.EndAcquisition()
                        except Exception:
                            pass
                finally:
                    try:
                        self._cam.DeInit()
                    except Exception:
                        pass
        finally:
            # Remove references before clearing/ releasing system
            self._nodemap = None
            self._nodemap_tldevice = None
            # Clear camera list (if available)
            try:
                if self._cam_list is not None:
                    self._cam_list.Clear()
            except Exception:
                pass
            # Drop camera reference
            try:
                self._cam = None
            except Exception:
                pass
            # Release system last
            try:
                if self._system is not None:
                    self._system.ReleaseInstance()
            except Exception:
                # Ignore Spinnaker interface release errors on shutdown
                pass
            self._system = None
            self._cam_list = None
            self._running = False
