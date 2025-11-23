from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from ..pipeline import Frame, Plugin


_CMAP_NAME_TO_CODE = {
    'inferno': getattr(cv2, 'COLORMAP_INFERNO', cv2.COLORMAP_JET),
    'jet': cv2.COLORMAP_JET,
    'turbo': getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET),
    'hot': cv2.COLORMAP_HOT,
    'parula': getattr(cv2, 'COLORMAP_PARULA', cv2.COLORMAP_JET),
}


class TemperatureScalePlugin(Plugin):
    """Overlay a temperature colorbar on the displayed frame when Celsius data is available.

    Options:
        - dynamic: if True, per-frame min/max are estimated (robust percentiles).
        - min_c, max_c: fixed range (used when dynamic=False or when provided).
        - width: bar width in pixels.
        - position: 'right' or 'left'.
        - cmap: one of inferno|turbo|jet|hot|parula (fallback to JET if unavailable).
        - num_ticks: number of tick labels to draw including endpoints.
    """

    def __init__(
        self,
        *,
        dynamic: bool = True,
        min_c: Optional[float] = None,
        max_c: Optional[float] = None,
        low_percentile: float = 2.0,
        high_percentile: float = 98.0,
        width: int = 24,
        margin: int = 8,
        position: str = 'right',
        cmap: str = 'inferno',
        num_ticks: int = 5,
        alpha: float = 0.95,
        font_scale: float = 0.3,
        apply_cmap_to_frame: bool = True,
        overlay_preserve_s_threshold: int = 40,
        show_colorbar: bool = False,
    ) -> None:
        self.name = 'temp_scale'
        self.dynamic = bool(dynamic)
        self.fixed_min = min_c
        self.fixed_max = max_c
        self.low_pct = float(low_percentile)
        self.high_pct = float(high_percentile)
        self.width = int(max(8, width))
        self.margin = int(max(0, margin))
        self.position = 'right' if str(position).lower() not in ('left',) else 'left'
        self.cmap_code = _CMAP_NAME_TO_CODE.get(str(cmap).lower(), cv2.COLORMAP_JET)
        self.num_ticks = max(2, int(num_ticks))
        self.alpha = float(max(0.0, min(1.0, alpha)))
        self.font_scale = float(max(0.3, font_scale))
        self.apply_cmap_to_frame = bool(apply_cmap_to_frame)
        self._overlay_s_thresh = int(max(0, min(255, overlay_preserve_s_threshold)))
        self.show_colorbar = bool(show_colorbar)

    def process(self, frame: Frame, ctx: Dict[str, object]) -> None:
        if frame.celsius is None or frame.bgr is None:
            return

        H, W = frame.bgr.shape[:2]
        c = frame.celsius.astype(np.float32)
        # Compute range
        if not self.dynamic and self.fixed_min is not None and self.fixed_max is not None and self.fixed_max > self.fixed_min:
            cmin, cmax = float(self.fixed_min), float(self.fixed_max)
        else:
            # robust percentiles to avoid outliers
            cmin = float(np.nanpercentile(c, self.low_pct))
            cmax = float(np.nanpercentile(c, self.high_pct))
            # allow fixed override if provided
            if self.fixed_min is not None:
                cmin = float(self.fixed_min)
            if self.fixed_max is not None:
                cmax = float(self.fixed_max)
            if cmax <= cmin:
                cmax = cmin + 1e-3

        # Optionally colorize the frame to match the colormap (align visuals)
        if self.apply_cmap_to_frame:
            # Normalize full-frame celsius to 0..255 using same cmin/cmax
            norm_full = np.clip((c - cmin) / (cmax - cmin), 0.0, 1.0)
            lut_in_full = (norm_full * 255.0).astype(np.uint8)
            colored = cv2.applyColorMap(lut_in_full, self.cmap_code)
            # Preserve prior overlays drawn by earlier plugins by keeping high-saturation pixels
            hsv = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            mask = (sat >= self._overlay_s_thresh)
            # Composite: colored base, put original overlay pixels where mask is true
            out = colored.copy()
            out[mask] = frame.bgr[mask]
            frame.bgr = out

        if self.show_colorbar:
            # Build colorbar image (top=hot=cmax, bottom=cold=cmin) with height H and width
            grad = np.linspace(cmax, cmin, H, dtype=np.float32).reshape(H, 1)
            norm = np.clip((grad - cmin) / (cmax - cmin), 0.0, 1.0)
            lut_in = (norm * 255.0).astype(np.uint8)
            bar = np.repeat(lut_in, self.width, axis=1)
            bar_color = cv2.applyColorMap(bar, self.cmap_code)

            # Draw ticks and labels
            ticks = np.linspace(0, H - 1, self.num_ticks).astype(int)
            vals = np.linspace(cmax, cmin, self.num_ticks)
            # Prepare label canvas next to bar if needed (for readability outside frame)
            label_pad = 6
            # Side-by-side composition: create a new canvas that docks the colorbar beside the video
            spacer = np.full((H, self.margin, 3), 0, dtype=np.uint8) if self.margin > 0 else None
            if self.position == 'right':
                parts = [frame.bgr]
                if spacer is not None:
                    parts.append(spacer)
                parts.append(bar_color)
                composed = cv2.hconcat(parts)
                # Draw ticks on the composed image at x positions corresponding to bar
                x0 = frame.bgr.shape[1] + (self.margin if spacer is not None else 0)
            else:
                parts = [bar_color]
                if spacer is not None:
                    parts.append(spacer)
                parts.append(frame.bgr)
                composed = cv2.hconcat(parts)
                x0 = 0

            # Draw tick lines and labels alongside the bar on composed image
            for ty, val in zip(ticks, vals):
                # Tick line over bar edge
                cv2.line(composed, (x0 - 3, int(ty)), (x0, int(ty)), (255, 255, 255), 1)
                label = f"{val:.1f}°C"
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
                tx = x0 - label_pad - text_size[0]
                ty_txt = int(np.clip(ty + text_size[1] // 2, 0, H - 1))
                cv2.putText(composed, label, (tx, ty_txt), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 0, 0), 2, cv2.LINE_8)
                cv2.putText(composed, label, (tx, ty_txt), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 1, cv2.LINE_8)

            frame.bgr = composed

        # Store current range in context for optional logging
        ctx[self.name] = {"min_c": cmin, "max_c": cmax}
