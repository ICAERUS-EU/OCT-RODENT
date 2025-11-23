"""Export thermal raw streams into individual 16-bit frame files.

The script scans `results/raw` for files named `output_XXX.raw` (000-030 by default)
and writes each frame to `<stem>_<frame_idx>.raw` using the original 16-bit dtype.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def discover_raw_files(directory: Path, start: int, end: int) -> Iterable[Path]:
    """Yield raw file paths between the given indices if they exist."""
    for idx in range(start, end + 1):
        candidate = directory / f"output_{idx:03d}.raw"
        if candidate.exists():
            yield candidate
        else:
            print(f"[skip] Missing {candidate}")


def load_stream(raw_path: Path) -> tuple[np.ndarray, dict]:
    """Load raw thermal data and return the reshaped frame array and metadata."""
    meta_path = raw_path.with_suffix(raw_path.suffix + ".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file missing for {raw_path}: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    width = int(meta["width"])
    height = int(meta["height"])
    dtype = np.dtype(meta["written_dtype"])
    frame_count = int(meta.get("frame_count", 0))
    if frame_count <= 0:
        raise ValueError(f"No frames reported in metadata for {raw_path}")

    raw = np.fromfile(raw_path, dtype=dtype)
    expected = frame_count * height * width
    if raw.size != expected:
        raise ValueError(
            f"Raw stream size {raw.size} does not match expected {expected} for {raw_path}"
        )

    frames = raw.reshape(frame_count, height, width)
    return frames, meta


def export_frames(raw_path: Path, destination: Path) -> None:
    """Save each frame of the raw stream as an individual 16-bit .raw file."""
    frames, meta = load_stream(raw_path)
    destination.mkdir(parents=True, exist_ok=True)

    stem = raw_path.stem
    dtype = np.dtype(meta["written_dtype"])
    for idx, frame in enumerate(frames):
        out_path = destination / f"{stem}_{idx:05d}.raw"
        frame.astype(dtype, copy=False).tofile(out_path)
    print(f"[done] {stem}: exported {len(frames)} frames to {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save frames from raw thermal streams as individual 16-bit files.",
    )
    parser.add_argument(
        "--input-dir",
        default="results/raw",
        help="Directory containing output_XXX.raw files (default: results/raw)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/raw/frames",
        help="Destination directory for extracted frames (default: results/raw/frames)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting raw index (inclusive, default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=30,
        help="Ending raw index (inclusive, default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for raw_file in discover_raw_files(input_dir, args.start, args.end):
        target_dir = output_dir / raw_file.stem
        export_frames(raw_file, target_dir)


if __name__ == "__main__":
    main()
