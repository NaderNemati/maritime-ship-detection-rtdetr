#!/usr/bin/env python3
import argparse, os, sys, json, glob
from pathlib import Path
from PIL import Image
import yaml

# --- YOLO -> COCO helpers (from your code, simplified into one file) ---
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

def _gather_images(split_root, image_subdirs):
    paths = []
    for sub in image_subdirs:
        p = os.path.join(split_root, sub)
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(p, f"*{ext}")))
    return sorted(paths)

def _label_for_image(img_abs, split_root):
    rel = os.path.relpath(img_abs, split_root)
    if "/images/" in rel:
        rel_lbl = rel.replace("/images/", "/labels/")
    elif rel.startswith("images/"):
        rel_lbl = rel.replace("images/", "labels/")
    elif rel.startswith("gan/images/"):
        rel_lbl = rel.replace("gan/images/", "gan/labels/")
    else:
        rel_noext = os.path.splitext(rel)[0]
        rel_lbl = rel_noext + ".txt"
        if not os.path.exists(os.path.join(split_root, rel_lbl)):
            return None
        return os.path.join(split_root, rel_lbl)
    base_noext = os.path.splitext(os.path.basename(rel))[0]
    rel_lbl = os.path.join(os.path.dirname(rel_lbl), base_noext + ".txt")
    return os.path.join(split_root, rel_lbl)

def yolo_to_coco(split_root, image_subdirs, out_json_path, class_names):
    images, annotations = [], []
    ann_id, img_id = 1, 1
    img_paths = _gather_images(split_root, image_subdirs)

    for img_path in img_paths:
        rel_img = os.path.relpath(img_path, split_root)
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception as e:
            print(f"Skipping unreadable image: {img_path} ({e})")
            continue

        images.append({"id": img_id, "file_name": rel_img, "width": W, "height": H})

        lbl_path = _label_for_image(img_path, split_root)
        if lbl_path and os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    c, xc, yc, w, h = map(float, parts)
                    c = int(c)
                    x = (xc - w/2.0) * W
                    y = (yc - h/2.0) * H
                    bw = w * W
                    bh = h * H
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": c + 1,  # COCO ids start at 1
                        "bbox": [x, y, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    ann_id += 1
        img_id += 1

    categories = [{"id": i+1, "name": name} for i, name in enumerate(class_names)]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(coco, f)
    return len(images), len(annotations)

# --- YAML builder (from your code, folded here) ---
def build_data_yaml(data_root, classes_file, out_path="tdss.yaml", include_gan=False):
    data_root = os.path.abspath(data_root)
    with open(classes_file, "r") as f:
        names = yaml.safe_load(f).get("names", [])
    if not names:
        raise ValueError("classes_file must contain 'names: [...]'")

    train_paths = [f"{data_root}/train/images"]
    if include_gan and os.path.isdir(f"{data_root}/train/gan/images"):
        train_paths.append(f"{data_root}/train/gan/images")

    data_yaml = {
        "path": data_root,
        "train": train_paths if len(train_paths) > 1 else train_paths[0],
        "val": f"{data_root}/valid/images",
        "test": f"{data_root}/test/images",
        "names": names
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    return out_path

# --- COCO patcher (from your code) ---
def patch_coco_supercategory(json_path, supercategory="marine"):
    with open(json_path, "r") as f:
        coco = json.load(f)
    for c in coco.get("categories", []):
        c.setdefault("supercategory", supercategory)
    with open(json_path, "w") as f:
        json.dump(coco, f)

# --- Train/val/predict using Ultralytics RT-DETR ---
def train_rtdetr(data_yaml, weights="rtdetr-l.pt", epochs=100, imgsz=640, batch=8, device="auto",
                 optimizer="AdamW", lr0=1e-4, cos_lr=True, patience=20, amp=False, mode="train"):
    from ultralytics import YOLO
    model = YOLO(weights)
    if mode == "train":
        model.train(
            data=data_yaml,
            device=device,
            imgsz=imgsz,
            epochs=epochs,
            batch=batch,
            workers=0,
            optimizer=optimizer,
            lr0=lr0,
            cos_lr=cos_lr,
            patience=patience,
            amp=amp,
        )
    else:
        model.val(data=data_yaml, device=device, imgsz=imgsz)

def predict(weights, source, device="auto", imgsz=640, conf=0.25, save=True):
    from ultralytics import YOLO
    model = YOLO(weights)
    _ = model.predict(source=source, device=device, imgsz=imgsz, conf=conf, save=save)

# --- CLI ---
def main():
    ap = argparse.ArgumentParser(description="All-in-one pipeline for RT-DETR ship detection")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s_yaml = sub.add_parser("build-yaml")
    s_yaml.add_argument("--data_root", required=True)
    s_yaml.add_argument("--classes_file", default="configs/classes.yaml")
    s_yaml.add_argument("--out", default="tdss.yaml")
    s_yaml.add_argument("--include_gan", action="store_true")

    s_y2c = sub.add_parser("yolo2coco")
    s_y2c.add_argument("--split_root", required=True)
    s_y2c.add_argument("--classes_file", default="configs/classes.yaml")
    s_y2c.add_argument("--out", required=True)
    s_y2c.add_argument("--image_subdirs", nargs="+", default=["images"])

    s_patch = sub.add_parser("patch-coco")
    s_patch.add_argument("--json_path", required=True)
    s_patch.add_argument("--supercategory", default="marine")

    s_train = sub.add_parser("train")
    s_train.add_argument("--data", required=True)
    s_train.add_argument("--weights", default="rtdetr-l.pt")
    s_train.add_argument("--epochs", type=int, default=100)
    s_train.add_argument("--imgsz", type=int, default=640)
    s_train.add_argument("--batch", type=int, default=8)
    s_train.add_argument("--device", default="auto")
    s_train.add_argument("--optimizer", default="AdamW")
    s_train.add_argument("--lr0", type=float, default=1e-4)
    s_train.add_argument("--cos_lr", action="store_true")
    s_train.add_argument("--patience", type=int, default=20)
    s_train.add_argument("--amp", action="store_true")

    s_val = sub.add_parser("val")
    s_val.add_argument("--data", required=True)
    s_val.add_argument("--weights", required=True)
    s_val.add_argument("--imgsz", type=int, default=640)
    s_val.add_argument("--device", default="auto")

    s_pred = sub.add_parser("predict")
    s_pred.add_argument("--weights", required=True)
    s_pred.add_argument("--source", required=True)
    s_pred.add_argument("--device", default="auto")
    s_pred.add_argument("--imgsz", type=int, default=640)
    s_pred.add_argument("--conf", type=float, default=0.25)
    s_pred.add_argument("--save", action="store_true")

    s_qs = sub.add_parser("quickstart", help="One-shot: yaml -> coco -> patch -> train -> predict")
    s_qs.add_argument("--data_root", required=True)
    s_qs.add_argument("--classes_file", default="configs/classes.yaml")
    s_qs.add_argument("--include_gan", action="store_true")
    s_qs.add_argument("--epochs", type=int, default=100)
    s_qs.add_argument("--imgsz", type=int, default=640)
    s_qs.add_argument("--batch", type=int, default=8)
    s_qs.add_argument("--device", default="auto")
    s_qs.add_argument("--weights", default="rtdetr-l.pt")
    s_qs.add_argument("--make_coco", action="store_true", help="Also create & patch COCO JSONs")

    args = ap.parse_args()

    if args.cmd == "build-yaml":
        out = build_data_yaml(args.data_root, args.classes_file, args.out, args.include_gan)
        print(f"Wrote data YAML -> {out}")

    elif args.cmd == "yolo2coco":
        with open(args.classes_file, "r") as f:
            names = yaml.safe_load(f).get("names", [])
        if not names:
            raise ValueError("classes_file must define 'names'")
        ni, na = yolo_to_coco(os.path.abspath(args.split_root), args.image_subdirs,
                              os.path.abspath(args.out), names)
        print(f"[COCO] root={args.split_root} images={ni} ann={na} -> {args.out}")

    elif args.cmd == "patch-coco":
        patch_coco_supercategory(args.json_path, args.supercategory)
        print(f"Patched supercategory='{args.supercategory}' in {args.json_path}")

    elif args.cmd == "train":
        train_rtdetr(args.data, args.weights, args.epochs, args.imgsz, args.batch,
                     args.device, args.optimizer, args.lr0, args.cos_lr, args.patience, args.amp, mode="train")

    elif args.cmd == "val":
        # simple wrapper using train()'s val mode
        train_rtdetr(args.data, args.weights, epochs=0, imgsz=args.imgsz, batch=1,
                     device=args.device, optimizer="AdamW", lr0=1e-4, cos_lr=False,
                     patience=0, amp=False, mode="val")

    elif args.cmd == "predict":
        predict(args.weights, args.source, args.device, args.imgsz, args.conf, args.save)
        print("Saved predictions under runs/detect/predict")

    elif args.cmd == "quickstart":
        data_yaml = build_data_yaml(args.data_root, args.classes_file, "tdss.yaml", args.include_gan)
        if args.make_coco:
            # Train split: include real (+gan if present)
            subs = ["images"]
            if args.include_gan and os.path.isdir(os.path.join(args.data_root, "train/gan/images")):
                subs.append("gan/images")
            for split, subs_ in [("train", subs), ("valid", ["images"]), ("test", ["images"])]:
                out_json = os.path.join(args.data_root, split, "_annotations.coco.json")
                with open(args.classes_file, "r") as f:
                    names = yaml.safe_load(f).get("names", [])
                yolo_to_coco(os.path.join(args.data_root, split), subs_, out_json, names)
                patch_coco_supercategory(out_json)

        train_rtdetr(data_yaml, args.weights, args.epochs, args.imgsz, args.batch, args.device,
                     optimizer="AdamW", lr0=1e-4, cos_lr=True, patience=20, amp=False, mode="train")

        predict(weights="runs/detect/train/weights/best.pt",
                source=os.path.join(args.data_root, "test/images"),
                device=args.device, imgsz=args.imgsz, conf=0.25, save=True)
        print("Quickstart done. Check runs/detect/*")

if __name__ == "__main__":
    # Uncomment to force CPU-only:
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()

