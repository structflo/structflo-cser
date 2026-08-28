"""Dump raw YOLO (ultralytics) detections at a low conf floor for offline, backend-agnostic eval.

Run this ONCE while ultralytics is still installed. Output JSON per split:
  {"meta": {...}, "images": {stem: {"file", "width", "height", "ms", "dets": [{"bbox", "conf", "class_id"}]}}}
Boxes are xyxy in ORIGINAL image pixels. Images are converted to grayscale-RGB exactly as
ChemPipeline.detect() does, and run full-image at imgsz=1280 (the deployed regime).
"""
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = {
    "real_test": "data/finetune/yolo/real_test/images",
    "real_val": "data/finetune/yolo/real_val/images",
    "synth_test": "data/generated_test/val/images",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/labels_detect/finetune_plus/weights/best.pt")
    ap.add_argument("--out-dir", default="runs/license_migration/preds/yolo_v0.4")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--scale", type=float, default=1.0,
                    help="pre-downscale images by this factor (0.48 ≈ 144 dpi, 0.5 ≈ 150 dpi renders of 300-dpi pages); "
                         "boxes are mapped back to original pixels so the same labels apply")
    args = ap.parse_args()

    from ultralytics import YOLO

    sha = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
    model = YOLO(args.weights)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        img_dir = Path(SPLITS[split])
        paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        images: dict[str, dict] = {}
        t_start = time.perf_counter()
        for i, p in enumerate(paths):
            im = Image.open(p).convert("L").convert("RGB")
            run_im = im
            if args.scale != 1.0:
                run_im = im.resize((max(1, round(im.width * args.scale)), max(1, round(im.height * args.scale))), Image.Resampling.LANCZOS)
            arr = np.array(run_im)
            t0 = time.perf_counter()
            res = model(arr, conf=args.conf, imgsz=args.imgsz, verbose=False, max_det=args.max_det)[0]
            ms = (time.perf_counter() - t0) * 1000
            sx, sy = im.width / run_im.width, im.height / run_im.height
            dets = []
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                dets.append({"bbox": [x1 * sx, y1 * sy, x2 * sx, y2 * sy], "conf": float(box.conf[0]), "class_id": int(box.cls[0])})
            images[p.stem] = {"file": p.name, "width": im.width, "height": im.height, "ms": ms, "dets": dets}
            if (i + 1) % 100 == 0:
                print(f"[{split}] {i + 1}/{len(paths)}", flush=True)
        payload = {
            "meta": {
                "backend": "ultralytics", "weights": args.weights, "sha256": sha,
                "conf_floor": args.conf, "imgsz": args.imgsz, "max_det": args.max_det,
                "grayscale": True, "tile": False, "scale": args.scale, "n_images": len(paths),
                "wall_s": time.perf_counter() - t_start,
            },
            "images": images,
        }
        (out_dir / f"{split}.json").write_text(json.dumps(payload))
        med = float(np.median([v["ms"] for v in images.values()])) if images else 0.0
        print(f"[{split}] wrote {len(paths)} images, median {med:.1f} ms/img → {out_dir / f'{split}.json'}", flush=True)


if __name__ == "__main__":
    main()
