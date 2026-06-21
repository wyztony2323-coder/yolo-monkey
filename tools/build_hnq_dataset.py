"""Build a crop-based HNQ candidate dataset.

The dataset is intentionally external to YOLO training:
  - does not modify YOLO labels
  - uses frozen YOLO predictions and GT jitter boxes as proposals
  - optionally adds hard-negative video candidates (human/black-shadow/glare)

Output:
  out_dir/
    crops/*.jpg
    candidates.csv
    report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

FINALMODEL = Path(__file__).resolve().parents[1]
ULOCAL = FINALMODEL / "ultralytics_local"
sys.path.insert(0, str(ULOCAL))
sys.path.insert(0, str(FINALMODEL))

import ultralytics_rainforest  # noqa: F401
from ultralytics import YOLO

VIDEO_DIR = Path(r"D:\college\college3\monkey\用于测试的视频")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_DATA = FINALMODEL / "dataset_split_631" / "ordered" / "split_1" / "data.yaml"
DEFAULT_WEIGHTS = FINALMODEL / "result_p2_respam_p2only_ft800" / "fold_1" / "weights" / "best.pt"


def read_yaml_min(path: Path) -> dict:
    """Small YAML reader for simple Ultralytics data.yaml files."""
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip().strip("'\"")
    return data


def resolve_split_dir(data_yaml: Path, split: str) -> Path:
    d = read_yaml_min(data_yaml)
    root = Path(d.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    p = Path(d[split])
    return p if p.is_absolute() else (root / p)


def yolo_label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_gt(img_path: Path, w: int, h: int) -> np.ndarray:
    lp = yolo_label_path(img_path)
    boxes = []
    if not lp.exists():
        return np.zeros((0, 4), dtype=np.float32)
    for line in lp.read_text(encoding="utf-8").splitlines():
        p = line.strip().split()
        if len(p) < 5:
            continue
        _, cx, cy, bw, bh = map(float, p[:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    aa = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    bb = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def delta_target(box: np.ndarray, gt: np.ndarray) -> list[float]:
    bx = (box[0] + box[2]) / 2
    by = (box[1] + box[3]) / 2
    bw = max(1.0, box[2] - box[0])
    bh = max(1.0, box[3] - box[1])
    gx = (gt[0] + gt[2]) / 2
    gy = (gt[1] + gt[3]) / 2
    gw = max(1.0, gt[2] - gt[0])
    gh = max(1.0, gt[3] - gt[1])
    return [
        float((gx - bx) / bw),
        float((gy - by) / bh),
        float(math.log(gw / bw)),
        float(math.log(gh / bh)),
    ]


def apply_clip(box: np.ndarray, w: int, h: int) -> np.ndarray:
    b = box.astype(np.float32).copy()
    b[[0, 2]] = np.clip(b[[0, 2]], 0, w - 1)
    b[[1, 3]] = np.clip(b[[1, 3]], 0, h - 1)
    if b[2] <= b[0] + 1:
        b[2] = min(w - 1, b[0] + 2)
    if b[3] <= b[1] + 1:
        b[3] = min(h - 1, b[1] + 2)
    return b


def jitter_box(gt: np.ndarray, w: int, h: int) -> np.ndarray:
    bw = gt[2] - gt[0]
    bh = gt[3] - gt[1]
    cx = (gt[0] + gt[2]) / 2 + random.uniform(-0.15, 0.15) * bw
    cy = (gt[1] + gt[3]) / 2 + random.uniform(-0.15, 0.15) * bh
    scale = random.uniform(0.8, 1.2)
    aspect = random.uniform(0.9, 1.1)
    nw = bw * scale * math.sqrt(aspect)
    nh = bh * scale / math.sqrt(aspect)
    return apply_clip(np.array([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2], dtype=np.float32), w, h)


def random_background_box(w: int, h: int) -> np.ndarray:
    """Sample a background proposal from a no-target frame."""
    area = random.uniform(0.003, 0.06) * w * h
    aspect = random.uniform(0.6, 1.8)
    bw = min(w * 0.8, math.sqrt(area * aspect))
    bh = min(h * 0.8, math.sqrt(area / aspect))
    cx = random.uniform(bw / 2, max(bw / 2, w - bw / 2))
    cy = random.uniform(bh / 2, max(bh / 2, h - bh / 2))
    return apply_clip(np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dtype=np.float32), w, h)


def geom_vec(box: np.ndarray, score: float, W: int, H: int) -> list[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    area = (bw * bh) / float(W * H)
    aspect = bw / bh
    if area < 0.015:
        hid, stride = 0, 4
    elif area < 0.06:
        hid, stride = 1, 8
    elif area < 0.20:
        hid, stride = 2, 16
    else:
        hid, stride = 3, 32
    onehot = [1.0 if i == hid else 0.0 for i in range(4)]
    return [
        ((x1 + x2) / 2) / W,
        ((y1 + y2) / 2) / H,
        bw / W,
        bh / H,
        area,
        min(aspect, 10.0) / 10.0,
        float(score),
        *onehot,
        stride / 32.0,
    ]


def crop_box(img: np.ndarray, box: np.ndarray, context: float = 1.25) -> np.ndarray:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box.astype(float)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * context, (y2 - y1) * context
    x1 = int(max(0, cx - bw / 2))
    y1 = int(max(0, cy - bh / 2))
    x2 = int(min(W - 1, cx + bw / 2))
    y2 = int(min(H - 1, cy + bh / 2))
    return img[y1 : y2 + 1, x1 : x2 + 1].copy()


def avg_hash(img: np.ndarray, size: int = 8) -> np.ndarray:
    """Tiny perceptual hash for removing repeated video crops."""
    if img.size == 0:
        return np.zeros((size * size,), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).astype(np.uint8).reshape(-1)


def hash_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def cv_imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def image_files(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)


def add_row(rows, crop_root: Path, img: np.ndarray, img_path: Path, source: str, idx: int, box, score, label, q, delta, split: str):
    crop = crop_box(img, box)
    if crop.size == 0:
        return
    crop_name = f"{source}_{Path(img_path).stem}_{idx:04d}.jpg"
    crop_path = crop_root / crop_name
    if not cv_imwrite_unicode(crop_path, crop):
        return
    H, W = img.shape[:2]
    g = geom_vec(box, score, W, H)
    rows.append(
        {
            "crop_path": str(crop_path),
            "image_path": str(img_path),
            "source": source,
            "split": split,
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
            "score_yolo": float(score),
            "label": int(label),
            "q_box": float(q),
            "tx": float(delta[0]),
            "ty": float(delta[1]),
            "tw": float(delta[2]),
            "th": float(delta[3]),
            **{f"g{i}": float(v) for i, v in enumerate(g)},
        }
    )


def build_split(model, data_yaml: Path, split: str, args, crop_root: Path) -> list[dict]:
    rows = []
    imgs = image_files(resolve_split_dir(data_yaml, split))
    random.shuffle(imgs)
    if args.max_images > 0:
        imgs = imgs[: args.max_images]
    for n, img_path in enumerate(imgs, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = load_gt(img_path, W, H)
        ridx = 0
        # GT jitter positives
        for gi, g in enumerate(gt):
            for _ in range(args.jitter_per_gt):
                jb = jitter_box(g, W, H)
                q = float(iou_matrix(jb[None, :], g[None, :])[0, 0])
                add_row(rows, crop_root, img, img_path, f"{split}_gt", ridx, jb, 1.0, 1, q, delta_target(jb, g), split)
                ridx += 1

        # Model proposals
        res = model.predict(img, conf=args.low_conf, iou=args.iou, max_det=args.max_det, verbose=False, device=args.device)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.cpu().numpy().astype(float)
        if len(gt):
            ious = iou_matrix(boxes, gt)
            best_iou = ious.max(axis=1)
            best_gt = ious.argmax(axis=1)
        else:
            best_iou = np.zeros((len(boxes),), dtype=np.float32)
            best_gt = np.zeros((len(boxes),), dtype=np.int64)
        pos_idx = [i for i, v in enumerate(best_iou) if v >= args.pos_iou]
        neg_idx = [i for i, v in enumerate(best_iou) if v < args.neg_iou]
        random.shuffle(pos_idx)
        random.shuffle(neg_idx)
        pos_idx = pos_idx[: args.max_pos_pred]
        neg_idx = neg_idx[: args.max_neg_pred]
        for i in pos_idx:
            g = gt[best_gt[i]]
            add_row(rows, crop_root, img, img_path, f"{split}_pred_pos", ridx, boxes[i], scores[i], 1, best_iou[i], delta_target(boxes[i], g), split)
            ridx += 1
        for i in neg_idx:
            add_row(rows, crop_root, img, img_path, f"{split}_pred_neg", ridx, boxes[i], scores[i], 0, 0.0, [0, 0, 0, 0], split)
            ridx += 1
        if n % 50 == 0:
            print(f"[{split}] {n}/{len(imgs)} rows={len(rows)}", flush=True)
    return rows


def parse_video_specs(text: str):
    """Parse '10:195:205,11:0:-1' -> [(idx,start,end), ...]."""
    specs = []
    if not text:
        return specs
    for part in text.split(","):
        p = part.strip().split(":")
        if len(p) != 3:
            continue
        specs.append((int(p[0]), float(p[1]), float(p[2])))
    return specs


def build_hard_videos(model, args, crop_root: Path) -> list[dict]:
    """Build video hard negatives from manually confirmed no-target clips only.

    Important: this function has no ground-truth labels for videos. Therefore
    every clip passed through --hard_video_specs must be a no-gibbon segment.
    Use it for human/background/glare false positives, not for target videos.
    """
    rows = []
    videos = sorted(f for f in os.listdir(VIDEO_DIR) if Path(f).suffix.lower() in VIDEO_EXTS)
    for vidx, start, end in parse_video_specs(args.hard_video_specs):
        vname = videos[vidx - 1]
        cap = cv2.VideoCapture(str(VIDEO_DIR / vname))
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        f0 = int(max(0, start) * fps)
        f1 = total - 1 if end < 0 else min(total - 1, int(end * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        raw = f0
        kept = 0
        accepted = 0
        seen_hashes: list[np.ndarray] = []
        while raw <= f1:
            ok, frame = cap.read()
            if not ok:
                break
            if (raw - f0) % args.hard_video_stride == 0:
                frame_added = 0
                res = model.predict(frame, conf=args.low_conf, iou=args.iou, max_det=args.max_det, verbose=False, device=args.device)[0]
                if res.boxes is not None and len(res.boxes):
                    boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
                    scores = res.boxes.conf.cpu().numpy().astype(float)
                    cand = []
                    H, W = frame.shape[:2]
                    for box, score in zip(boxes, scores):
                        area = max(0.0, float((box[2] - box[0]) * (box[3] - box[1]) / (W * H)))
                        if score < args.hard_video_min_conf or score > args.hard_video_max_conf:
                            continue
                        if area < args.hard_video_min_area or area > args.hard_video_max_area:
                            continue
                        crop = crop_box(frame, box)
                        h = avg_hash(crop)
                        if any(hash_distance(h, old) <= args.hard_video_hash_thresh for old in seen_hashes):
                            continue
                        cand.append((box, score, h))
                    cand = sorted(cand, key=lambda x: x[1], reverse=True)[: args.max_hard_per_frame]
                    for ci, (box, score, h) in enumerate(cand):
                        if args.max_hard_total > 0 and accepted >= args.max_hard_total:
                            break
                        pseudo = Path(f"{Path(vname).stem}_f{raw:06d}")
                        add_row(rows, crop_root, frame, pseudo, "video_hard_neg", ci, box, score, 0, 0.0, [0, 0, 0, 0], "train")
                        seen_hashes.append(h)
                        accepted += 1
                        frame_added += 1
                if args.hard_video_background_per_frame > 0:
                    H, W = frame.shape[:2]
                    for bi in range(args.hard_video_background_per_frame):
                        if args.max_hard_total > 0 and accepted >= args.max_hard_total:
                            break
                        if frame_added >= args.max_hard_per_frame:
                            break
                        box = random_background_box(W, H)
                        crop = crop_box(frame, box)
                        h = avg_hash(crop)
                        if any(hash_distance(h, old) <= args.hard_video_hash_thresh for old in seen_hashes):
                            continue
                        pseudo = Path(f"{Path(vname).stem}_f{raw:06d}_bg")
                        add_row(rows, crop_root, frame, pseudo, "video_bg_neg", bi, box, 0.0, 0, 0.0, [0, 0, 0, 0], "train")
                        seen_hashes.append(h)
                        accepted += 1
                        frame_added += 1
                kept += 1
            if args.max_hard_total > 0 and accepted >= args.max_hard_total:
                break
            raw += 1
        cap.release()
        print(f"[hard] video#{vidx} {vname} frames={kept} accepted={accepted} rows={len(rows)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--out_dir", default=str(FINALMODEL / "hnq_dataset"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--low_conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.75)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--pos_iou", type=float, default=0.5)
    ap.add_argument("--neg_iou", type=float, default=0.3)
    ap.add_argument("--jitter_per_gt", type=int, default=3)
    ap.add_argument("--max_pos_pred", type=int, default=32)
    ap.add_argument("--max_neg_pred", type=int, default=64)
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--clean_out", action="store_true", help="remove existing out_dir before rebuilding")
    ap.add_argument("--hard_video_specs", default="", help="manual no-gibbon clips only, e.g. 11:0:-1")
    ap.add_argument("--hard_video_stride", type=int, default=5)
    ap.add_argument("--max_hard_per_frame", type=int, default=5)
    ap.add_argument("--max_hard_total", type=int, default=300, help="cap accepted video hard negatives per video")
    ap.add_argument("--hard_video_min_conf", type=float, default=0.05)
    ap.add_argument("--hard_video_max_conf", type=float, default=1.0)
    ap.add_argument("--hard_video_min_area", type=float, default=0.0002)
    ap.add_argument("--hard_video_max_area", type=float, default=0.25)
    ap.add_argument("--hard_video_hash_thresh", type=int, default=6, help="smaller means stricter visual dedupe")
    ap.add_argument("--hard_video_background_per_frame", type=int, default=0, help="deduped random background crops from no-target clips")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.clean_out and out_dir.exists():
        shutil.rmtree(out_dir)
    crop_root = out_dir / "crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    data_yaml = Path(args.data)
    model = YOLO(args.weights)
    random.seed(42)
    np.random.seed(42)

    rows = []
    for split in ("train", "val"):
        rows.extend(build_split(model, data_yaml, split, args, crop_root))
    rows.extend(build_hard_videos(model, args, crop_root))

    csv_path = out_dir / "candidates.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
    report = {
        "rows": len(rows),
        "positive": sum(int(r["label"]) for r in rows),
        "negative": sum(1 - int(r["label"]) for r in rows),
        "sources": {},
    }
    for r in rows:
        report["sources"][r["source"]] = report["sources"].get(r["source"], 0) + 1
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] HNQ dataset: {out_dir}")


if __name__ == "__main__":
    main()
