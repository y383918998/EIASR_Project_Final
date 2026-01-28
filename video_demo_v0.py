"""Command-line demo for the lane and vehicle detection prototype."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2

from .config import PipelineConfig
from .pipeline import LaneVehiclePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lane detection + vehicle distance prototype")
    parser.add_argument("--video", type=Path, required=True, help="Path to input video")
    parser.add_argument(
        "--svm", type=Path, default=None, help="Optional path to linear SVM model for vehicles"
    )
    parser.add_argument(
        "--focal",
        type=float,
        default=None,
        help="Camera focal length in pixels (used for distance estimation)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show video frames interactively (press q to stop)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write an annotated video (MP4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig()
    if args.svm is not None:
        config.vehicles.svm_model_path = str(args.svm)
    if args.focal is not None:
        config.vehicles.focal_length_px = args.focal

    pipeline = LaneVehiclePipeline(config)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    writer: Optional[cv2.VideoWriter] = None
    if args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay, lanes, vehicles = pipeline.process_frame(frame)

        if writer is not None:
            writer.write(overlay)

        if args.display:
            cv2.imshow("Lane + Vehicle Prototype", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()