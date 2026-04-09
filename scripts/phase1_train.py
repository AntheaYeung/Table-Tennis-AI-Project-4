"""Train a YOLOv8/YOLO11 model for Phase 1 table-tennis detection.

Phase 1 targets:
- ball
- table
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 1 detector (ball + table).")
    parser.add_argument("--data", required=True, help="Path to YOLO dataset YAML file.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model checkpoint.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="0", help="CUDA device id (e.g. 0) or 'cpu'.")
    parser.add_argument("--project", default="runs/phase1", help="Output project folder.")
    parser.add_argument("--name", default="ball_table_detector", help="Run name.")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
