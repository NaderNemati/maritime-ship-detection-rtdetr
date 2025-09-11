# cpu_train_tuned.py
import os, cv2, torch
from ultralytics import YOLO

# --- CPU threading ---
N = os.cpu_count() or 4
torch.set_num_threads(N)          # math kernels
torch.set_num_interop_threads(1)  # dispatcher threads (keep small)
torch.backends.mkldnn.enabled = True
try:
    cv2.setNumThreads(0)          # let MKL/OpenMP own the threads
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

DATA = "/home/nader/Desktop/Ship_Detection/mini_ship_data/subset.yaml"
WEIGHT = "/home/nader/maritime-ship-detection-rtdetr/maritime-ship-detection-rtdetr/runs/detect/train3/weights/best.pt"  # or "rtdetr-l.pt"

WORKERS = min(max(2, N // 2), 8)  # good starting point on CPU

model = YOLO(WEIGHT)
results = model.train(
    data=DATA,
    device="cpu",
    workers=WORKERS,
    batch=max(2, N // 4),
    imgsz=512,          # use 640 if you can tolerate slower speed
    epochs=10,          # increase when you have time
    rect=True,          # less padding -> fewer CPU cycles
    cache="ram",        # if you have free RAM; else "disk" or disable
    amp=False,          # AMP not useful on CPU
    cos_lr=True,
    optimizer="AdamW",
    lr0=0.005,
    weight_decay=5e-4,
    mosaic=0.5,         # lighter augs to reduce CPU load
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.1,
)
print("Saved to:", getattr(results, "save_dir", None))

