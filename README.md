# Table Tennis AI Project (Phase 1)

This project helps beginners build **Phase 1** of a table-tennis AI system:

1. Detect the **table tennis ball**.
2. Detect the **table**.

The code uses **YOLOv8** (`ultralytics`) and supports `.mov` videos.

---

## 0) What you need first

- Python 3.10+ recommended
- A YOLO-format dataset already prepared
- Your dataset should include exactly these class names:
  - `ball`
  - `table`

If your class names are different, edit the scripts accordingly.

---

## 1) Setup your environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2) Prepare dataset YAML

Create a file like `data/table_tennis_phase1.yaml`:

```yaml
path: /absolute/path/to/your/dataset
train: images/train
val: images/val

names:
  0: ball
  1: table
```

> Folder layout example:
>
> ```
> dataset/
>   images/
>     train/
>     val/
>   labels/
>     train/
>     val/
> ```

---

## 3) Train Phase 1 model

```bash
python scripts/phase1_train.py \
  --data data/table_tennis_phase1.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch 8 \
  --project runs/phase1 \
  --name ball_table_detector
```

After training, your best weights are usually here:

```text
runs/phase1/ball_table_detector/weights/best.pt
```

---

## 4) Run inference on a `.mov` video

```bash
python scripts/phase1_infer_video.py \
  --weights runs/phase1/ball_table_detector/weights/best.pt \
  --video /absolute/path/to/input.mov \
  --output outputs/phase1_annotated.mp4 \
  --csv outputs/phase1_positions.csv
```

This generates:

- Annotated video with `ball` and `table` boxes
- CSV with frame-level detection info (useful for Phase 2)

---

## 5) Beginner tips to improve ball detection

- Use high resolution (e.g., `imgsz=1280`)
- Ensure many frames where the ball is small/fast
- Add motion-blur examples in training data
- Keep labels very accurate for the ball
- Try larger models (`yolo11s.pt`, `yolo11m.pt`) if GPU allows

---

## 6) Next (Phase 2 preview)

In Phase 2, you will:

- Track player movement (pose or person tracking)
- Combine with ball positions from Phase 1 CSV
- Build event logic (hit timing, bounce, rally stats)

For now, complete and validate Phase 1 first.
