
<div align="center">
  <table border=0 style="border: 0px solid #c6c6c6 !important; border-spacing: 0px; width: auto !important;">
    <tr>
      <td valign=top style="border: 0px solid #c6c6c6 !important; padding: 0px !important;">
        <div align=center valign=top>
          <img src="https://github.com/NaderNemati/maritime-ship-detection-rtdetr/blob/main/detr.png" style="margin: 0px !important; height: 400px !important;">
        </div>
      </td>
    </tr>
  </table>
</div>

---
# Maritime Ship Detection with RT-DETR (Real+Synthetic ➜ Real)
This repository provides a compact workflow for transformer-based ship detection using **Ultralytics RT-DETR**.
Given a YOLO-organized dataset including train/gan/ (synthetic) data and includes a single CLI (pipeline.py) to:

build Ultralytics data YAML,

(optionally) convert YOLO ↔ COCO and patch categories,

train/validate RT-DETR,

prediction and evaluation.


### Protocol

Training set size: Real + Synthetic — 3,781 train, 49 val, 50 test (val/test are real). 

Evaluation: mAP@0.5, tested on Real images.

Schedule: 300 epochs, default RT-DETR loss/assignment, Ultralytics dataloader/augs.

Notes: If your raw dataset differs, the subset builder below reproduces the same split sizes (199 real train + 3,582 synthetic train = 3,781).



## Repository Structure
```python
ship-detection-rtdetr/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ configs/
│  └─ classes.yaml
├─ eval_tuned.py
└─ train_tuned.py

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

## Class list

```yaml
names:
  - motor_boat
  - sailing_boat
  - seamark
```



## Installation

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

Python 3.9–3.11 recommended

Ultralytics will auto-download rtdetr-*.pt weights on first use




## requirements.txt

```bash
torch>=2.1
ultralytics>=8.2.0
pillow
tqdm
pyyaml
numpy
opencv-python
onnx>=1.15
onnxruntime>=1.18

```

#### Create a matching split (Real+Synthetic → Real) with 3,781 train, 49 val, 50 test:

```bash

python subset_yolo_dataset.py \
  --src /path/to/DATA_ROOT \
  --dst /path/to/DATA_MATCHED \
  --per_train 199 \
  --per_val 49 \
  --per_test 50 \
  --include_gan --per_train_gan 3582 \
  --require_label --min_anns 1 \
  --seed 7 \
  --classes_yaml configs/classes.yaml \
  --out_yaml tdss.yaml \
  --link_mode copy
```

#### This writes /path/to/DATA_MATCHED/tdss.yaml with:

```yaml

path: /path/to/DATA_MATCHED
train:
  - train/images
  - train/gan/images     # included because --include_gan
val: valid/images         # real
test: test/images         # real
names: [motor_boat, sailing_boat, seamark]

```
TIP: If your gan/labels are incomplete, the script will still include images but requires labels for training.

#### Training (300 epochs, Real+Synthetic ➜ Real)

Train RT-DETR-L for 300 epochs on the matched split:

```bash
python pipeline.py train \
  --data /path/to/DATA_MATCHED/tdss.yaml \
  --weights rtdetr-l.pt \
  --epochs 300 \
  --imgsz 640 \
  --batch 16 \
  --device auto     # GPU if available, else CPU

```
TIPS:

For small/far vessels, consider --imgsz 768–1024 (requires more VRAM).

If using a CPU-only setup, reduce --batch and --imgsz to keep training feasible.


### Evaluation (mAP@0.5 on Real test set)

Evaluate your best checkpoint on the Real test with mAP@0.5:

```bash
python pipeline.py val \
  --data /path/to/DATA_MATCHED/tdss.yaml \
  --weights runs/detect/train/weights/best.pt \
  --imgsz 640 \
  --split test
```

PR curves and confusion matrices are saved under runs/detect/val*


### Predict
Save labeled predictions on test images:


```bash
python pipeline.py predict \
  --weights runs/detect/train/weights/best.pt \
  --source /path/to/DATA_MATCHED/test/images \
  --imgsz 640 --conf 0.25 --save

```

### Results

|            Scenario             |     mAP\@0.5    |        Precision       |        Recall       |      F1      |
| ------------------------------- | :-------------: | :--------------------: | :-----------------: |  :--------:  |
|        **Real+Syn → Real**      |     **0.88**    |          0.91          |         0.90        |   **0.89**   |



### RT-DETR-L Model Summary

|  # | from          |  n |    params | module                                    | arguments                             |
| -: | :------------ | -: | --------: | :---------------------------------------- | :------------------------------------ |
|  0 | -1            |  1 |    25,248 | ultralytics.nn.modules.block.HGStem       | \[3, 32, 48]                          |
|  1 | -1            |  6 |   155,072 | ultralytics.nn.modules.block.HGBlock      | \[48, 48, 128, 3, 6]                  |
|  2 | -1            |  1 |     1,408 | ultralytics.nn.modules.conv.DWConv        | \[128, 128, 3, 2, 1, False]           |
|  3 | -1            |  6 |   839,296 | ultralytics.nn.modules.block.HGBlock      | \[128, 96, 512, 3, 6]                 |
|  4 | -1            |  1 |     5,632 | ultralytics.nn.modules.conv.DWConv        | \[512, 512, 3, 2, 1, False]           |
|  5 | -1            |  6 | 1,695,360 | ultralytics.nn.modules.block.HGBlock      | \[512, 192, 1024, 5, 6, True, False]  |
|  6 | -1            |  6 | 2,055,808 | ultralytics.nn.modules.block.HGBlock      | \[1024, 192, 1024, 5, 6, True, True]  |
|  7 | -1            |  6 | 2,055,808 | ultralytics.nn.modules.block.HGBlock      | \[1024, 192, 1024, 5, 6, True, True]  |
|  8 | -1            |  1 |    11,264 | ultralytics.nn.modules.conv.DWConv        | \[1024, 1024, 3, 2, 1, False]         |
|  9 | -1            |  6 | 6,708,480 | ultralytics.nn.modules.block.HGBlock      | \[1024, 384, 2048, 5, 6, True, False] |
| 10 | -1            |  1 |   524,800 | ultralytics.nn.modules.conv.Conv          | \[2048, 256, 1, 1, None, 1, 1, False] |
| 11 | -1            |  1 |   789,760 | ultralytics.nn.modules.transformer.AIFI   | \[256, 1024, 8]                       |
| 12 | -1            |  1 |    66,048 | ultralytics.nn.modules.conv.Conv          | \[256, 256, 1, 1]                     |
| 13 | -1            |  1 |         0 | torch.nn.modules.upsampling.Upsample      | \[None, 2, 'nearest']                 |
| 14 | 7             |  1 |   262,656 | ultralytics.nn.modules.conv.Conv          | \[1024, 256, 1, 1, None, 1, 1, False] |
| 15 | \[-2, -1]     |  1 |         0 | ultralytics.nn.modules.conv.Concat        | \[1]                                  |
| 16 | -1            |  3 | 2,232,320 | ultralytics.nn.modules.block.RepC3        | \[512, 256, 3]                        |
| 17 | -1            |  1 |    66,048 | ultralytics.nn.modules.conv.Conv          | \[256, 256, 1, 1]                     |
| 18 | -1            |  1 |         0 | torch.nn.modules.upsampling.Upsample      | \[None, 2, 'nearest']                 |
| 19 | 3             |  1 |   131,584 | ultralytics.nn.modules.conv.Conv          | \[512, 256, 1, 1, None, 1, 1, False]  |
| 20 | \[-2, -1]     |  1 |         0 | ultralytics.nn.modules.conv.Concat        | \[1]                                  |
| 21 | -1            |  3 | 2,232,320 | ultralytics.nn.modules.block.RepC3        | \[512, 256, 3]                        |
| 22 | -1            |  1 |   590,336 | ultralytics.nn.modules.conv.Conv          | \[256, 256, 3, 2]                     |
| 23 | \[-1, 17]     |  1 |         0 | ultralytics.nn.modules.conv.Concat        | \[1]                                  |
| 24 | -1            |  3 | 2,232,320 | ultralytics.nn.modules.block.RepC3        | \[512, 256, 3]                        |
| 25 | -1            |  1 |   590,336 | ultralytics.nn.modules.conv.Conv          | \[256, 256, 3, 2]                     |
| 26 | \[-1, 12]     |  1 |         0 | ultralytics.nn.modules.conv.Concat        | \[1]                                  |
| 27 | -1            |  3 | 2,232,320 | ultralytics.nn.modules.block.RepC3        | \[512, 256, 3]                        |
| 28 | \[21, 24, 27] |  1 | 7,308,017 | ultralytics.nn.modules.head.RTDETRDecoder | \[3, \[256, 256, 256]]                |


rt-detr-l summary: 457 layers, 32,812,241 params, 108.0 GFLOPs


### Practical Tips

Small/far targets: increase --imgsz or tile inference; keep batch reasonable.

Synthetic quality: include only synthetic images with correct labels.

CPU-only: use --device cpu, --batch 2–4, --imgsz 512–640; expect longer training.

Class order: YAML names must align with label IDs across all splits.

COCO export: optional for cross-tool eval/analysis; PR curves help diagnose class imbalance.













