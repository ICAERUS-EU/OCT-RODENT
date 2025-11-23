from __future__ import annotations
from ultralytics import YOLO

import argparse
import os

from framework.pipeline import Pipeline
from framework.sources.opencv_source import OpenCVSource
from framework.sources.raw_file_source import RawFileSource

try:
    from framework.sources.spinnaker_source import SpinnakerSource  # noqa: F401
    HAS_SPINNAKER = True
except Exception:
    HAS_SPINNAKER = False

from framework.plugins.yolo_plugin import YOLOPlugin
from framework.plugins.opencv_blob_plugin import MedianBackgroundBlobPlugin
from framework.plugins.temperature_scale_plugin import TemperatureScalePlugin

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Modular image acquisition + plugin processing pipeline for rodent thermal detection")

    # Source options
    p.add_argument("--source", choices=["opencv", "spinnaker"], default="spinnaker", help="Frame source backend (spinnaker for FLIR real-time, opencv for video/files)")
    p.add_argument("--input", default="results/raw/output_000.raw", help="OpenCV input: camera index, video path, image path, folder, or raw file/metadata (ignored for spinnaker). Chooses either OpenCV source or RawFileSource based on extension.")
    p.add_argument("--input-meta", default="", help="Optional metadata JSON path for raw inputs (default: input + .json or metadata 'path')")

    # Spinnaker source options
    p.add_argument("--camera-index", type=int, default=0, help="Spinnaker camera index (default: 0)")

    # OpenCV source options
    p.add_argument("--width", type=int, default=320, help="Resize width (OpenCV source)")
    p.add_argument("--height", type=int, default=256, help="Resize height (OpenCV source)")

    # Playback options
    p.add_argument("--max-fps", type=float, default=0.0, help="Maximum playback FPS for prerecorded sources (0 = unlimited)")

    # Visualization/colormap for thermal frames
    p.add_argument("--temp-scale", type=int, default=1, help="Colorize thermal frames (1/0)")
    p.add_argument("--temp-bar", type=int, default=0, help="Show temperature colorbar (1/0)")
    p.add_argument("--cmap", choices=["inferno", "turbo", "jet", "hot", "parula"], default="inferno", help="Colormap for thermal visualization")

    # Recording/output
    p.add_argument("--save", type=int, default=0, help="Save annotated output video (1/0)")
    p.add_argument("--save-path", default="results/mp4/output_000.mp4", help="Output video file path (.mp4/.avi)")
    p.add_argument("--save-fps", type=float, default=20.0, help="Output video FPS (e.g., 30; set explicitly for correct playback speed)")

    # Raw thermal saving
    p.add_argument("--raw-save", type=int, default=0, help="Save raw thermal stream with 14-bit values (1/0)")
    p.add_argument("--raw-path", default="results/raw/output_000.raw", help="Raw thermal output file path (.raw/.bin)")
    p.add_argument("--raw-meta", default="", help="Optional metadata JSON path (default: raw-path + .json)")

    # Overlay options
    p.add_argument("--display", type=int, default=1, help="Show window (1) or headless (0)")
    p.add_argument("--global-overlay", type=int, default=1, help="Enable drawing of overlays (1/0) (including boxes and labels)")
    p.add_argument("--show-count", type=int, default=0, help="Show object count overlay (1/0)")
    p.add_argument("--fps", type=int, default=0, help="Show FPS overlay (1/0)")
    p.add_argument("--temp-overlay", type=int, default=0, help="Show min/max temperature overlay (1/0)")

    # Object counting reset options
    p.add_argument("--count-reset-frames", type=int, default=10, help="Frames with zero detections before count resets to 0 (0 to disable)")
    p.add_argument("--count-reset-sec", type=float, default=0.0, help="Seconds with zero detections before count resets to 0 (0 to disable)")

    # YOLO plugin options
    p.add_argument("--yolo", type=int, default=0, help="Enable YOLO plugin (1/0)")
    p.add_argument("--model", default="models/yolo11n-finetuned-best.pt", help="YOLO model path")
    p.add_argument("--conf", type=float, default=0.8, help="YOLO confidence threshold")
    p.add_argument("--device", default="cpu", help="YOLO device: cpu/cuda")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO input size")
    p.add_argument("--yolo-nms-iou", type=float, default=0.25, help="IoU threshold passed to Ultralytics predict() NMS")

    # YOLO multi-scale options
    p.add_argument("--yolo-ms", type=int, default=0, help="Enable multi-scale inference (1/0)")
    p.add_argument("--yolo-ms-scales", default="1.0,2.5", help="Comma-separated scales, e.g. '1.0,1.5,2.0'")
    p.add_argument("--yolo-ms-score", type=float, default=0.25, help="Score threshold for multi-scale merge")
    p.add_argument("--yolo-ms-iou", type=float, default=0.45, help="IoU threshold for NMS in multi-scale merge")
    p.add_argument("--yolo-ms-per-class-nms", type=int, default=1, help="Run per-class NMS (1/0)")

    # Blob detection plugin
    p.add_argument("--blob", type=int, default=0, help="Enable median background blob plugin (1/0)")
    p.add_argument("--blob-buffer", type=int, default=25, help="Buffer size (frames) for median background tracking")
    p.add_argument("--blob-min-area", type=int, default=125, help="Min blob area in pixels")
    p.add_argument("--blob-max-area", type=int, default=1750, help="Max blob area in pixels (0 disables)")

    # Relative thresholding options
    p.add_argument("--blob-rel-mode", choices=["off", "hot", "cold", "both"], default="hot", help="Relative-to-background thresholding mode")
    p.add_argument("--blob-delta-c", type=float, default=2.5, help="Delta in Celsius for relative thresholding")

    # Motion detection options
    p.add_argument("--blob-combine", choices=["union", "intersect"], default="intersect", help="Combine motion and relative masks")
    p.add_argument("--blob-motion-delta-c", type=float, default=2.5, help="Celsius delta for motion in thermal domain")

    # Static hot-spot detection
    p.add_argument("--blob-hot-static", type=int, default=1, help="Enable spatial hot-spot detection (1/0)")
    p.add_argument("--blob-hot-kernel", type=int, default=5, help="Kernel size (odd) for local blur in hot-spot detection")
    p.add_argument("--blob-hot-delta-c", type=float, default=5.0, help="Celsius delta vs local baseline for hot-spot")

    # Noise reduction and tracking
    p.add_argument("--blob-noise-kernel", type=int, default=7, help="Gaussian kernel size for pre-diff smoothing (odd, <=1 disables)")
    p.add_argument("--blob-persist-frames", type=int, default=10, help="Minimum frames to persist before reporting a blob")
    p.add_argument("--blob-persist-sec", type=float, default=0.25, help="Minimum seconds to persist before reporting (0 to disable)")
    p.add_argument("--blob-track-max-miss", type=int, default=5, help="Max consecutive misses before dropping a track")
    p.add_argument("--blob-track-iou", type=float, default=0.3, help="IoU threshold for associating detections to tracks")
    p.add_argument("--blob-static-after-frames", type=int, default=50, help="Frames a track must be static before relabeling moving->static")
    p.add_argument("--blob-moving-after-frames", type=int, default=5, help="Frames a track must be moving before relabeling static->moving")
    p.add_argument("--blob-static-pos-px", type=int, default=1, help="Max center movement (px) between frames to treat as static")
    p.add_argument("--blob-static-area-frac", type=float, default=0.1, help="Max relative area change to treat as static (e.g., 0.10 = 10%)")
    p.add_argument("--blob-static-iou", type=float, default=0.65, help="Minimum IoU between consecutive boxes to treat as static")

    # Static blob area filtering
    p.add_argument("--blob-static-min-area", type=int, default=0, help="Min area in pixels for static blobs (0 = use multiplier)")
    p.add_argument("--blob-static-min-area-mult", type=float, default=1.0, help="Multiplier on --blob-min-area for static blobs when explicit static min area is 0")
    
    # Blob weak-motion relabeling controls
    p.add_argument("--blob-moving-coverage-max", type=float, default=0.1, help="Max moving mask coverage to still consider weak motion (<= this => weak)")
    p.add_argument("--blob-hot-overlap-min", type=float, default=0.9, help="Min hot-spot overlap to relabel weak moving as static (>= this => static)")
    
    return p.parse_args()


def make_source(args):
    if args.source == "spinnaker":
        if not HAS_SPINNAKER:
            raise RuntimeError("Spinnaker/PySpin not available; install and ensure camera is connected")
        return SpinnakerSource(camera_index=args.camera_index, auto_configure=True)
    else:
        inp = args.input
        meta_override = args.input_meta.strip() if isinstance(getattr(args, "input_meta", ""), str) else ""

        if isinstance(inp, str):
            lower = inp.lower()
            # Raw file playback (.raw/.bin) or metadata JSON
            if lower.endswith(('.raw', '.bin')) or lower.endswith('.json'):
                raw_path = inp if lower.endswith(('.raw', '.bin')) else None
                meta_path = meta_override or (inp if lower.endswith('.json') else None)
                override_fps = args.max_fps if args.max_fps and args.max_fps > 0 else None
                return RawFileSource(raw_path, meta_path=meta_path, loop=False, override_fps=override_fps)

        # Try to parse camera index if input is digit-like
        if isinstance(inp, str) and inp.isdigit():
            inp2 = int(inp)
        else:
            inp2 = inp
        resize = (args.width, args.height) if args.width and args.height else None
        return OpenCVSource(inp2, loop=True, resize=resize)


def make_plugins(args):
    plugins = []
    overlays_enabled = bool(getattr(args, "global_overlay", 1))
    # Apply temperature colormap and colorbar first so later plugins can draw on top
    if args.temp_scale:
        plugins.append(
            TemperatureScalePlugin(
                cmap=args.cmap,
                apply_cmap_to_frame=True,
                show_colorbar=bool(args.temp_bar) and overlays_enabled,
            )
        )
    if args.yolo:
        # Parse scales list
        try:
            ms_scales = [float(s.strip()) for s in str(args.yolo_ms_scales).split(',') if s.strip()]
        except Exception:
            ms_scales = [1.0]
        plugins.append(
            YOLOPlugin(
                model_path=args.model,
                conf=args.conf,
                device=args.device,
                imgsz=args.imgsz,
                nms_iou=args.yolo_nms_iou,
                draw=overlays_enabled,
                multi_scale=bool(args.yolo_ms),
                ms_scales=ms_scales,
                ms_score_thresh=args.yolo_ms_score,
                ms_iou_thresh=args.yolo_ms_iou,
                ms_per_class_nms=bool(args.yolo_ms_per_class_nms),
            )
        )
    if args.blob:
        plugins.append(
            MedianBackgroundBlobPlugin(
                buffer_size=args.blob_buffer,
                min_area=args.blob_min_area,
                draw=overlays_enabled,
                label="object",
                rel_mode=args.blob_rel_mode,
                delta_c=args.blob_delta_c,
                combine=args.blob_combine,
                motion_delta_c=args.blob_motion_delta_c,
                # spatial hot-spot
                hot_static=bool(args.blob_hot_static),
                hot_kernel=args.blob_hot_kernel,
                hot_delta_c=args.blob_hot_delta_c,
                noise_kernel=args.blob_noise_kernel,
                # area filtering (pixels)
                max_area=args.blob_max_area,
                static_min_area=args.blob_static_min_area,
                static_min_area_mult=args.blob_static_min_area_mult,
                persist_min_frames=args.blob_persist_frames,
                persist_min_seconds=args.blob_persist_sec,
                track_max_miss=args.blob_track_max_miss,
                track_iou_thresh=args.blob_track_iou,
                static_after_frames=args.blob_static_after_frames,
                moving_after_frames=args.blob_moving_after_frames,
                static_pos_px=args.blob_static_pos_px,
                static_area_frac=args.blob_static_area_frac,
                static_iou=args.blob_static_iou,
                # weak-motion relabeling
                moving_coverage_max=args.blob_moving_coverage_max,
                hot_overlap_min=args.blob_hot_overlap_min,
            )
        )
    return plugins


def main():
    args = build_args()
    source = make_source(args)
    plugins = make_plugins(args)
    overlays_enabled = bool(getattr(args, "global_overlay", 1))
    raw_path = args.raw_path if args.raw_save else None
    raw_meta = None
    if args.raw_save:
        meta_arg = getattr(args, "raw_meta", "")
        if isinstance(meta_arg, str):
            meta_arg = meta_arg.strip()
            raw_meta = meta_arg or None
    pipe = Pipeline(
        source,
        plugins,
        window_name=f"Pipeline - {args.source}",
        display=bool(args.display),
        show_fps=bool(args.fps),
        save_path=(args.save_path if args.save else None),
        save_fps=(args.save_fps if args.save_fps and args.save_fps > 0 else None),
        raw_path=raw_path,
        raw_meta_path=raw_meta,
        show_count=bool(args.show_count),
        count_reset_frames=args.count_reset_frames,
        count_reset_sec=args.count_reset_sec,
        max_playback_fps=(args.max_fps if args.max_fps and args.max_fps > 0 else None),
        show_temp_overlay=bool(args.temp_overlay),
        enable_overlays=overlays_enabled,
    )
    pipe.run()


if __name__ == "__main__":
    main()
