import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_frames(raw_path: Path) -> tuple[np.ndarray, dict]:
    """Load thermal frames from a raw file and convert to Celsius."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    new_suffix = raw_path.suffix + ".json" if raw_path.suffix else ".json"
    meta_path = raw_path.with_suffix(new_suffix)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    width = int(meta["width"])
    height = int(meta["height"])
    dtype = np.dtype(meta["written_dtype"])
    frame_count = int(meta.get("frame_count", 0))
    if frame_count <= 0:
        raise ValueError("No frames found in raw stream")

    raw = np.fromfile(raw_path, dtype=dtype)
    expected_size = frame_count * height * width
    if raw.size != expected_size:
        raise ValueError(
            f"Raw stream size {raw.size} does not match expected {expected_size}"
        )

    frames = raw.reshape(frame_count, height, width).astype(np.float32)
    frames = frames * 0.04 - 273.15  # Convert to Celsius
    return frames, meta


def launch_viewer(frames: np.ndarray, fps: float) -> None:
    """Display frames with click/keyboard navigation and timed playback."""
    frame_total = frames.shape[0]
    frame_idx = [0]
    is_playing = [False]

    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(frames[0], cmap="inferno")
    plt.colorbar(img, ax=ax, label="Temperature (°C)")
    ax.axis("off")
    title = ax.set_title("")

    def update_frame() -> None:
        idx = frame_idx[0]
        img.set_data(frames[idx])
        title.set_text(f"Thermal Frame {idx + 1}/{frame_total}")
        fig.canvas.draw_idle()

    def advance(step: int = 1) -> None:
        frame_idx[0] = (frame_idx[0] + step) % frame_total
        update_frame()

    timer_interval = max(int(1000 / max(fps, 1e-6)), 1)
    timer = fig.canvas.new_timer(interval=timer_interval)
    timer.add_callback(lambda: advance(1))

    def toggle_playback() -> None:
        if not is_playing[0]:
            timer.start()
            is_playing[0] = True
        else:
            timer.stop()
            is_playing[0] = False

    def on_key(event) -> None:
        # Allow P/space to toggle playback, arrows/A/D to seek, Q/Esc to quit.
        if event.key in {"p", " "}:
            toggle_playback()
        elif event.key in {"right", "d"}:
            advance(1)
        elif event.key in {"left", "a"}:
            advance(-1)
        elif event.key in {"escape", "q"}:
            plt.close(fig)

    def on_click(event) -> None:
        if event.button == 1:
            advance(1)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("close_event", lambda _event: timer.stop())

    update_frame()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive viewer for thermal raw streams converted to Celsius."
    )
    parser.add_argument(
        "--file",
        nargs="?",
        default="output_000",
        help="Path to the .raw file to visualize (default: output_000)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback frames per second when toggled (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = 'results/raw/' + args.file + '.raw'
    raw_path = Path(path)
    frames, _ = load_frames(raw_path)
    launch_viewer(frames, fps=max(args.fps, 0.1))


if __name__ == "__main__":
    main()