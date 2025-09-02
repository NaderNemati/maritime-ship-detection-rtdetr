

The repository implements a minimal end-to-end workflow for transformer-based ship detection. Given a YOLO-organized dataset (with optional train/gan/ images), pipeline.py can (a) build the Ultralytics data YAML, (b) optionally convert YOLO labels to COCO and patch categories, (c) fine-tune RT-DETR with sensible defaults, and (d) perform inference and save visualized predictions. Results (metrics, checkpoints, and predictions) are written under runs/. This serves as a strong, lightweight baseline that you can extend with tracking, AIS/Radar fusion, or uncertainty heads later.

## 📁 Repository Structure
```python
ship-detection-rtdetr/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ configs/
│  └─ classes.yaml
└─ pipeline.py          # single, all-in-one CLI

```

```python
DATA_ROOT/
  train/
    images/    labels/
    gan/
      images/  labels/    # optional
  valid/
    images/    labels/
  test/
    images/    labels/
```

