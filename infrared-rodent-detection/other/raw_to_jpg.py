from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Use a non-interactive backend so this script works in headless environments.
matplotlib.use("Agg", force=True)

FRAME_SHAPE = (256, 320)  # height, width
PIXEL_COUNT = FRAME_SHAPE[0] * FRAME_SHAPE[1]
DTYPE = np.dtype("<u2")  # unsigned 16-bit little-endian
TEMP_SCALE = 0.04
KELVIN_OFFSET = 273.15


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Convert FLIR raw frames to JPEG.")
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("rodent_dataset/dataset"),
		help="Directory with .raw frames.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("rodent_dataset/dataset_jpg"),
		help="Directory for generated .jpg files.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Regenerate JPG even if it already exists.",
	)
	return parser.parse_args()


def load_raw_frame(raw_path: Path) -> np.ndarray:
	buffer = np.fromfile(raw_path, dtype=DTYPE)
	if buffer.size != PIXEL_COUNT:
		raise ValueError(f"Unexpected frame size in {raw_path} (got {buffer.size} pixels)")
	return buffer.reshape(FRAME_SHAPE)


def raw_to_celsius(raw_frame: np.ndarray) -> np.ndarray:
	return raw_frame.astype(np.float32) * TEMP_SCALE - KELVIN_OFFSET


def save_temp_frame(temp_frame: np.ndarray, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.imsave(output_path, temp_frame, cmap="inferno")


def convert_directory(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
	raw_files = sorted(input_dir.glob("*.raw"))
	if not raw_files:
		print(f"No .raw files found in {input_dir}")
		return

	for raw_path in raw_files:
		output_path = output_dir / f"{raw_path.stem}.jpg"
		if output_path.exists() and not overwrite:
			continue

		raw_frame = load_raw_frame(raw_path)
		temp_frame = raw_to_celsius(raw_frame)
		save_temp_frame(temp_frame, output_path)


def main() -> None:
	args = parse_args()
	convert_directory(args.input, args.output, args.overwrite)


if __name__ == "__main__":
	main()
