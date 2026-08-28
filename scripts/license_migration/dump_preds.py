"""Dump raw D-FINE detections at a low conf floor (same JSON layout as dump_yolo_preds.py)."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from structflo.cser.inference.dfine import DFineDetector

SPLITS = {
    "real_test": "data/finetune/yolo/real_test/images",
    "real_val": "data/finetune/yolo/real_val/images",
    "synth_test": "data/generated_test/val/images",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--scale", type=float, default=1.0, help="pre-downscale factor (0.48 ≈ 144 dpi, 0.5 ≈ 150 dpi); boxes mapped back")
    args = ap.parse_args()

    sha = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
    det = DFineDetector.from_file(args.weights, imgsz=args.imgsz, amp=args.amp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        paths = sorted(p for p in Path(SPLITS[split]).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        images = {}
        t_start = time.perf_counter()
        for i, p in enumerate(paths):
            im = Image.open(p).convert("L").convert("RGB")
            run_im = im
            if args.scale != 1.0:
                run_im = im.resize((max(1, round(im.width * args.scale)), max(1, round(im.height * args.scale))), Image.Resampling.LANCZOS)
            arr = np.array(run_im)
            t0 = time.perf_counter()
            dets = det.predict(arr, conf=args.conf, imgsz=args.imgsz, max_det=args.max_det)
            ms = (time.perf_counter() - t0) * 1000
            if args.scale != 1.0:
                sx, sy = im.width / run_im.width, im.height / run_im.height
                for d in dets:
                    x1, y1, x2, y2 = d["bbox"]
                    d["bbox"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            images[p.stem] = {"file": p.name, "width": im.width, "height": im.height, "ms": ms, "dets": dets}
            if (i + 1) % 100 == 0:
                print(f"[{split}] {i + 1}/{len(paths)}", flush=True)
        payload = {
            "meta": {"backend": "dfine", "weights": args.weights, "sha256": sha, "conf_floor": args.conf,
                     "imgsz": args.imgsz, "max_det": args.max_det, "grayscale": True, "tile": False,
                     "amp": args.amp, "scale": args.scale, "n_images": len(paths), "wall_s": time.perf_counter() - t_start},
            "images": images,
        }
        (out_dir / f"{split}.json").write_text(json.dumps(payload))
        med = float(np.median([v["ms"] for v in images.values()])) if images else 0.0
        print(f"[{split}] wrote {len(paths)} images, median {med:.1f} ms/img → {out_dir / f'{split}.json'}", flush=True)


if __name__ == "__main__":
    main()
