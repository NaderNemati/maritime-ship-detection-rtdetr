This repository provides a compact workflow for transformer-based ship detection using Ultralytics RT-DETR.
Given a YOLO-organized dataset including train/gan/ images, the dtr_ship_detection.py script can build the Ultralytics data YAML, convert YOLO labels → COCO JSON and patch categories, and train / validate the RT-DETR model with sensible defaults as well as predict on test images and save visualized outputs.

This is a strong, lightweight baseline you can later extend with tracking, AIS/Radar fusion, or uncertainty heads. Furthermore, this pipeline contains flexible input, a standard YOLO layout, and GAN images mixed into training. COCO writes _annotations.coco.json per split + patches supercategory. In terms of computational resources, device auto uses GPU if available, otherwise CPU.


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

## Expected Dataset Layout

``` pgsql

DATA_ROOT/
  train/
    images/    labels/
    gan/
      images/  labels/    # optional (included only if --include_gan)
  valid/
    images/    labels/
  test/
    images/    labels/
```


## YOLO label format:

Update your classes in configs/classes.yaml:

```yaml
names:
  - motor_boat
  - sailing_boat
  - seamark

```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python: 3.9–3.11 recommended

Weights: Ultralytics will auto-download rtdetr-*.pt at first run

## Quickstart

```bash
python pipeline.py quickstart \
  --data_root /path/to/DATA_ROOT \
  --epochs 100 --imgsz 640 --batch 8 --device auto \
  --include_gan \
  --make_coco    # include this flag if you also want COCO JSONs generated & patched

```

This will:

Build tdss.yaml (includes train/gan/images if present and --include_gan is set)

(If --make_coco) create _annotations.coco.json for train/valid/test and set supercategory=marine

Train RT-DETR with reasonable defaults

Run predictions on test/images and save outputs under runs/detect/predict/


### Build Ultralytics data YAML

Creates tdss.yaml pointing to your splits and classes.

```bash
python pipeline.py build-yaml \
  --data_root /path/to/DATA_ROOT \
  --include_gan
```

### YOLO → COCO (optional)

Writes _annotations.coco.json under each split; patch-coco adds supercategory (default: marine).

```bash
python pipeline.py yolo2coco --split_root /path/to/DATA_ROOT/train --image_subdirs images gan/images
python pipeline.py yolo2coco --split_root /path/to/DATA_ROOT/valid --image_subdirs images
python pipeline.py yolo2coco --split_root /path/to/DATA_ROOT/test  --image_subdirs images

python pipeline.py patch-coco --json_path /path/to/DATA_ROOT/train/_annotations.coco.json
python pipeline.py patch-coco --json_path /path/to/DATA_ROOT/valid/_annotations.coco.json
python pipeline.py patch-coco --json_path /path/to/DATA_ROOT/test/_annotations.coco.json
```

### Train / Validate

1. Train from pre-trained RT-DETR weights (auto-downloaded by Ultralytics).

2. Validate an existing checkpoint.

```bash
python pipeline.py train --data tdss.yaml --weights rtdetr-l.pt --epochs 100 --imgsz 640 --batch 8 --device auto
python pipeline.py val   --data tdss.yaml --weights runs/detect/train/weights/best.pt --imgsz 640
```

### Predict

Runs inference on files/directories/URLs; saves labeled images.

```bash
python pipeline.py predict \
  --weights runs/detect/train/weights/best.pt \
  --source /path/to/DATA_ROOT/test/images \
  --imgsz 640 --conf 0.25 --save
```

### Practical Tips

1. Small/far targets: try --imgsz 1280 (or higher) and/or tile inference; keep batch reasonable.

2. Include GAN data: only if labels are good; enable via --include_gan.

3. CPU-only: run with --device cpu (or set CUDA_VISIBLE_DEVICES="" in pipeline.py).

4. Class order: must match configs/classes.yaml and your YOLO IDs.

5. COCO export: optional; helpful for cross-tool evaluation and analysis.


### Requirements

```txt
torch>=2.1
ultralytics>=8.2.0
pillow
tqdm
pyyaml
numpy
opencv-python
onnx>=1.15
onnxruntime>=1.18
# optional, if you later want RF-DETR/metrics
rfdetr>=1.2.0
rfdetr[metrics]

```

Happy shipping! 🚢














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


---

## requirements.txt

```txt
torch>=2.1
ultralytics>=8.2.0
pillow
tqdm
pyyaml
numpy
opencv-python
onnx>=1.15
onnxruntime>=1.18
# optional, if you later want RF-DETR/metrics
rfdetr>=1.2.0
rfdetr[metrics]

