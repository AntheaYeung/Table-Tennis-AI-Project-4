"""Run Phase 1 inference on a video and export detections.

Outputs:
1) Annotated video (.mp4)
2) CSV of frame-level positions for ball and table detections
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 inference on a .mov/.mp4 video.")
    parser.add_argument("--weights", required=True, help="Path to trained weights (best.pt).")
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--output", default="outputs/phase1_annotated.mp4", help="Output annotated video path.")
    parser.add_argument("--csv", default="outputs/phase1_positions.csv", help="Output CSV path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path = Path(args.output)
    csv_path = Path(args.csv)
    ensure_parent(output_path)
    ensure_parent(csv_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    class_names = model.names

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "frame_idx",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
        ])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)
            result = results[0]

            annotated = result.plot()
            writer.write(annotated)

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    cls_name = class_names.get(cls_id, str(cls_id))

                    csv_writer.writerow([
                        frame_idx,
                        cls_id,
                        cls_name,
                        round(conf, 4),
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                        round(cx, 2),
                        round(cy, 2),
                    ])

            frame_idx += 1

    cap.release()
    writer.release()

    print(f"Saved annotated video to: {output_path}")
    print(f"Saved detections CSV to: {csv_path}")


if __name__ == "__main__":
    main()
