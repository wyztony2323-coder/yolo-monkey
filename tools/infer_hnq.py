"""Run YOLO low-threshold candidates + crop-based HNQ late fusion."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

FINALMODEL = Path(__file__).resolve().parents[1]
ULOCAL = FINALMODEL / "ultralytics_local"
sys.path.insert(0, str(ULOCAL))
sys.path.insert(0, str(FINALMODEL))
sys.path.insert(0, str(FINALMODEL / "tools"))

import ultralytics_rainforest  # noqa: F401
from ultralytics import YOLO

from hnq_refiner import apply_delta_np, load_hnq, nms_np
from build_hnq_dataset import crop_box, geom_vec

try:
    from infer_videos_track_recover import TrackRecover
except Exception:
    TrackRecover = None

VIDEO_DIR = Path(r"D:\college\college3\monkey\用于测试的视频")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_YOLO = FINALMODEL / "result_p2_respam_p2only_ft800" / "fold_1" / "weights" / "best.pt"


def safe_name(stem: str, idx: int) -> str:
    ascii_part = "".join(c for c in stem if c.isascii() and (c.isalnum() or c in "-_"))[:30]
    return f"v{idx:02d}_{ascii_part}" if ascii_part else f"v{idx:02d}"


def collect_videos(indices: str, source: str):
    if source:
        p = Path(source)
        return [(1, p.name, p)]
    vids = sorted(f for f in os.listdir(VIDEO_DIR) if Path(f).suffix.lower() in VIDEO_EXTS)
    items = list(enumerate(vids, start=1))
    if indices:
        want = {int(x) for x in indices.split(",") if x.strip()}
        items = [(i, v) for i, v in items if i in want]
    return [(i, v, VIDEO_DIR / v) for i, v in items]


def prep_crops(img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, img_size: int, device: torch.device):
    crops, geoms = [], []
    H, W = img.shape[:2]
    for box, score in zip(boxes, scores):
        crop = crop_box(img, box)
        if crop.size == 0:
            crop = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        crops.append(crop)
        geoms.append(geom_vec(box, float(score), W, H))
    if not crops:
        return None, None
    x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255.0
    g = torch.tensor(np.asarray(geoms, dtype=np.float32))
    return x.to(device), g.to(device)


def hnq_refine_frame(yolo: YOLO, hnq, frame: np.ndarray, args, device: torch.device, score_min: float | None = None):
    res = yolo.predict(frame, conf=args.low_conf, iou=args.pre_iou, max_det=args.pre_topk, device=args.device, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
    boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = res.boxes.conf.cpu().numpy().astype(np.float32)
    x, geom = prep_crops(frame, boxes, scores, args.crop_size, device)
    if x is None:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
    with torch.no_grad():
        cls_logit, q_logit, delta = hnq(x, geom)
        p_gibbon = torch.sigmoid(cls_logit).view(-1).detach().cpu().numpy()
        q_box = torch.sigmoid(q_logit).view(-1).detach().cpu().numpy()
        delta_np = delta.detach().cpu().numpy()
    refined_boxes = apply_delta_np(boxes, delta_np, scale=args.delta_scale)
    H, W = frame.shape[:2]
    refined_boxes[:, [0, 2]] = np.clip(refined_boxes[:, [0, 2]], 0, W - 1)
    refined_boxes[:, [1, 3]] = np.clip(refined_boxes[:, [1, 3]], 0, H - 1)
    hnq_score = scores * p_gibbon * q_box
    final_scores = (1.0 - args.hnq_weight) * scores + args.hnq_weight * hnq_score
    score_min = args.final_conf if score_min is None else score_min
    keep = final_scores >= score_min
    boxes2 = refined_boxes[keep]
    scores2 = final_scores[keep]
    meta = [
        {"s_yolo": float(s), "p_gibbon": float(pg), "q_box": float(qb), "s_final": float(sf)}
        for s, pg, qb, sf, k in zip(scores, p_gibbon, q_box, final_scores, keep)
        if k
    ]
    if len(boxes2):
        keep_idx = nms_np(boxes2, scores2, iou_th=args.final_iou, topk=args.max_det)
        boxes2 = boxes2[keep_idx]
        scores2 = scores2[keep_idx]
        meta = [meta[i] for i in keep_idx]
    return boxes2, scores2, meta


def draw(frame, boxes, scores):
    out = frame.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0, 255, 0) if score >= 0.4 else (0, 215, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{score:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def draw_track(frame, outputs):
    out = frame.copy()
    for box, score, kind, tid in outputs:
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0, 255, 0) if kind == "high" else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{kind[0].upper()} id{tid} {score:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def run_video(yolo, hnq, args, device, idx: int, name: str, path: Path, out_root: Path):
    recover = None
    if args.track_recover:
        if TrackRecover is None:
            raise RuntimeError("TrackRecover import failed; check infer_videos_track_recover.py")
        recover = TrackRecover(
            match_iou=args.match_iou,
            center_ratio=args.center_ratio,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            max_miss=args.max_miss,
            init_min_hits=args.init_min_hits,
            recover_max_gap=args.recover_max_gap,
            recover_without_confirm_max=args.recover_without_confirm_max,
        )
        recover.reset()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"index": idx, "video": name, "error": "open_failed"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(np.ceil(total / args.max_keep))) if args.max_keep > 0 else 1
    out_fps = max(1.0, fps / stride)
    vdir = out_root / safe_name(Path(name).stem, idx)
    vdir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.save_video:
        suffix = "hnq_track" if args.track_recover else "hnq"
        writer = cv2.VideoWriter(str(vdir / f"{vdir.name}_{suffix}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))

    rows = []
    frames = frames_with_det = total_det = 0
    recovered_frames = recovered_total = 0
    conf_accum = []
    raw = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cur = raw
        raw += 1
        if cur % stride:
            continue
        score_min = args.recover_low_conf if args.track_recover else args.final_conf
        boxes, scores, meta = hnq_refine_frame(yolo, hnq, frame, args, device, score_min=score_min)
        outputs = None
        n_recovered = 0
        n_tracks = 0
        if recover is not None:
            high_mask = scores >= args.final_conf
            low_mask = (scores >= args.recover_low_conf) & (scores < args.final_conf)
            outputs, n_recovered, n_tracks = recover.step(boxes[high_mask], scores[high_mask], boxes[low_mask], scores[low_mask])
            boxes = np.asarray([o[0] for o in outputs], dtype=np.float32) if outputs else np.zeros((0, 4), dtype=np.float32)
            scores = np.asarray([o[1] for o in outputs], dtype=np.float32) if outputs else np.zeros((0,), dtype=np.float32)
        frames += 1
        if len(scores):
            frames_with_det += 1
            total_det += len(scores)
            conf_accum.extend(scores.tolist())
        if n_recovered:
            recovered_frames += 1
            recovered_total += n_recovered
        rows.append([
            frames - 1,
            cur,
            len(scores),
            float(scores.mean()) if len(scores) else 0.0,
            float(scores.max()) if len(scores) else 0.0,
            n_recovered,
            n_tracks,
        ])
        if writer is not None:
            writer.write(draw_track(frame, outputs) if outputs is not None else draw(frame, boxes, scores))
    cap.release()
    if writer is not None:
        writer.release()
    with open(vdir / "frame_stats.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["kept_idx", "raw_frame", "n_det", "mean_conf", "max_conf", "recovered", "active_tracks"])
        wr.writerows(rows)
    rec = {
        "index": idx,
        "video": name,
        "total_frames": total,
        "stride": stride,
        "frames_inferred": frames,
        "frames_with_det": frames_with_det,
        "total_det": total_det,
        "det_rate": round(frames_with_det / frames, 4) if frames else 0.0,
        "mean_conf": round(float(np.mean(conf_accum)), 4) if conf_accum else 0.0,
        "avg_det_per_frame": round(total_det / frames, 3) if frames else 0.0,
        "recovered_frames": recovered_frames,
        "recovered_total": recovered_total,
        "recover_rate": round(recovered_frames / frames, 4) if frames else 0.0,
    }
    (vdir / "summary.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_weights", default=str(DEFAULT_YOLO))
    ap.add_argument("--hnq_weights", required=True)
    ap.add_argument("--source", default="")
    ap.add_argument("--indices", default="1,2,7,10,11")
    ap.add_argument("--out_dir", default=str(FINALMODEL / "result_hnq_infer"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--low_conf", type=float, default=0.05)
    ap.add_argument("--pre_iou", type=float, default=0.75)
    ap.add_argument("--pre_topk", type=int, default=300)
    ap.add_argument("--final_conf", type=float, default=0.25)
    ap.add_argument("--final_iou", type=float, default=0.65)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--hnq_weight", type=float, default=0.3)
    ap.add_argument("--delta_scale", type=float, default=0.25)
    ap.add_argument("--crop_size", type=int, default=96)
    ap.add_argument("--max_keep", type=int, default=500)
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--track_recover", action="store_true")
    ap.add_argument("--recover_low_conf", type=float, default=0.05)
    ap.add_argument("--match_iou", type=float, default=0.25)
    ap.add_argument("--center_ratio", type=float, default=0.8)
    ap.add_argument("--scale_min", type=float, default=0.5)
    ap.add_argument("--scale_max", type=float, default=2.0)
    ap.add_argument("--max_miss", type=int, default=3)
    ap.add_argument("--init_min_hits", type=int, default=2)
    ap.add_argument("--recover_max_gap", type=int, default=2)
    ap.add_argument("--recover_without_confirm_max", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    yolo = YOLO(args.yolo_weights)
    hnq = load_hnq(args.hnq_weights, device=device)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for idx, name, path in collect_videos(args.indices, args.source):
        rec = run_video(yolo, hnq, args, device, idx, name, path, out_root)
        summary.append(rec)
        print(
            f"#{idx} {name} -> det_rate={rec.get('det_rate')} "
            f"recovered={rec.get('recovered_total')} mean_conf={rec.get('mean_conf')}",
            flush=True,
        )
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        keys = [
            "index",
            "video",
            "frames_inferred",
            "frames_with_det",
            "total_det",
            "det_rate",
            "mean_conf",
            "avg_det_per_frame",
            "recovered_frames",
            "recovered_total",
            "recover_rate",
        ]
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in summary:
            wr.writerow({k: r.get(k, "") for k in keys})
    print(f"[OK] HNQ inference: {out_root}")


if __name__ == "__main__":
    main()
