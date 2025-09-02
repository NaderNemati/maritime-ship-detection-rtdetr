

The repository implements a minimal end-to-end workflow for transformer-based ship detection. Given a YOLO-organized dataset (with optional train/gan/ images), pipeline.py can (a) build the Ultralytics data YAML, (b) optionally convert YOLO labels to COCO and patch categories, (c) fine-tune RT-DETR with sensible defaults, and (d) perform inference and save visualized predictions. Results (metrics, checkpoints, and predictions) are written under runs/. This serves as a strong, lightweight baseline that you can extend with tracking, AIS/Radar fusion, or uncertainty heads later.

## Repository Structure
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

## Quickstart
```bash
python pipeline.py quickstart \
  --data_root /path/to/DATA_ROOT \
  --epochs 100 --imgsz 640 --batch 8 --device auto \
  --include_gan

```


## Power-user subcommand


### Build Ultralytics data YAML
```bash
python pipeline.py build-yaml --data_root /path/to/DATA_ROOT --include_gan
```
### YOLO → COCO (optional)
```bash
python pipeline.py yolo2coco --split_root /path/to/DATA_ROOT/train --image_subdirs images gan/images
python pipeline.py yolo2coco --split_root /path/to/DATA_ROOT/valid --image_subdirs images
python pipeline.py yolo2coco --split_root /path/to/DATA_ROOT/test  --image_subdirs images
python pipeline.py patch-coco --json_path /path/to/DATA_ROOT/train/_annotations.coco.json
python pipeline.py patch-coco --json_path /path/to/DATA_ROOT/valid/_annotations.coco.json
python pipeline.py patch-coco --json_path /path/to/DATA_ROOT/test/_annotations.coco.json
```
### Train or Validate
```bash
python pipeline.py train --data tdss.yaml --weights rtdetr-l.pt --epochs 100 --imgsz 640 --batch 8 --device auto
python pipeline.py val   --data tdss.yaml --weights runs/detect/train/weights/best.pt --imgsz 640
```
### Predict
```bash
python pipeline.py predict --weights runs/detect/train/weights/best.pt --source /path/to/DATA_ROOT/test/images --imgsz 640 --conf 0.25 --save
```

