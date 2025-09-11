# cpu_eval_tuned.py
import os, cv2, torch
from ultralytics import YOLO

N = os.cpu_count() or 4
torch.set_num_threads(N)
torch.set_num_interop_threads(1)
try:
    cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

DATA = "/home/nader/Desktop/Ship_Detection/mini_ship_data/subset.yaml"
BEST = "/home/nader/maritime-ship-detection-rtdetr/maritime-ship-detection-rtdetr/runs/detect/train3/weights/best.pt"

WORKERS = min(max(2, N // 2), 8)
BATCH = max(2, N // 4)

model = YOLO(BEST)
for split in ("val", "test"):
    m = model.val(data=DATA, split=split, imgsz=512, batch=BATCH, device="cpu", workers=WORKERS, plots=True)
    print(f"[{split}] mAP50-95={m.box.map:.3f}  mAP50={m.box.map50:.3f}")

