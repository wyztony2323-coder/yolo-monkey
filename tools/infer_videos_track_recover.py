"""Detection-level temporal candidate recovery for YOLOv8s-P2 videos.

This tool does not modify the model. It runs YOLO at a low confidence threshold,
keeps high-confidence detections as normal outputs, and only recovers low-confidence
candidates when they match a short-term stable track prediction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
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
DEFAULT_WEIGHTS = FINALMODEL / "result_p2_respam_p2only_ft800" / "fold_1" / "weights" / "best.pt"
OUT_ROOT = FINALMODEL / "result_track_recover"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def xyxy_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    ba = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / (aa + ba - inter + 1e-9)


def center_distance_ok(a: np.ndarray, b: np.ndarray, ratio: float) -> bool:
    ac = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2], dtype=float)
    bc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], dtype=float)
    bw = max(1.0, float(b[2] - b[0]))
    bh = max(1.0, float(b[3] - b[1]))
    return float(np.linalg.norm(ac - bc)) <= ratio * math.sqrt(bw * bw + bh * bh)


def scale_ok(a: np.ndarray, b: np.ndarray, mn: float, mx: float) -> bool:
    aa = max(1.0, float(a[2] - a[0]) * float(a[3] - a[1]))
    ba = max(1.0, float(b[2] - b[0]) * float(b[3] - b[1]))
    r = aa / ba
    return mn <= r <= mx


@dataclass
class Track:
    tid: int
    box: np.ndarray
    conf: float
    hits: int = 1
    age: int = 0
    miss: int = 0
    recovered_streak: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))

    def predict(self) -> np.ndarray:
        return self.box + self.velocity

    def update(self, box: np.ndarray, conf: float, recovered: bool):
        new_v = box - self.box
        self.velocity = 0.7 * self.velocity + 0.3 * new_v
        self.box = box.astype(float)
        self.conf = float(conf)
        self.hits += 1
        self.age += 1
        self.miss = 0
        self.recovered_streak = self.recovered_streak + 1 if recovered else 0

    def decay_miss(self):
        self.box = self.predict()
        self.age += 1
        self.miss += 1


class TrackRecover:
    def __init__(
        self,
        match_iou: float,
        center_ratio: float,
        scale_min: float,
        scale_max: float,
        max_miss: int,
        init_min_hits: int,
        recover_max_gap: int,
        recover_without_confirm_max: int,
    ):
        self.match_iou = match_iou
        self.center_ratio = center_ratio
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.max_miss = max_miss
        self.init_min_hits = init_min_hits
        self.recover_max_gap = recover_max_gap
        self.recover_without_confirm_max = recover_without_confirm_max
        self.tracks: list[Track] = []
        self.next_id = 1

    def reset(self):
        self.tracks.clear()
        self.next_id = 1

    def _match(self, box: np.ndarray, used: set[int]) -> int | None:
        best_i = None
        best_score = -1.0
        for i, tr in enumerate(self.tracks):
            if i in used or tr.miss > self.max_miss:
                continue
            pred = tr.predict()
            iou = xyxy_iou(box, pred)
            if iou < self.match_iou and not center_distance_ok(box, pred, self.center_ratio):
                continue
            if not scale_ok(box, pred, self.scale_min, self.scale_max):
                continue
            score = iou
            if score > best_score:
                best_score = score
                best_i = i
        return best_i

    def step(self, high_boxes, high_confs, low_boxes, low_confs):
        outputs = []
        used_tracks: set[int] = set()
        used_low: set[int] = set()

        # High-confidence detections are always output and may create/update tracks.
        for box, conf in zip(high_boxes, high_confs):
            mi = self._match(box, used_tracks)
            if mi is None:
                tr = Track(self.next_id, box.astype(float), float(conf))
                self.next_id += 1
                self.tracks.append(tr)
                mi = len(self.tracks) - 1
            else:
                self.tracks[mi].update(box.astype(float), float(conf), recovered=False)
            used_tracks.add(mi)
            outputs.append((box, float(conf), "high", self.tracks[mi].tid))

        # Low-confidence candidates can only be recovered near existing stable tracks.
        for li, (box, conf) in enumerate(zip(low_boxes, low_confs)):
            mi = self._match(box, used_tracks)
            if mi is None:
                continue
            tr = self.tracks[mi]
            stable = tr.hits >= self.init_min_hits
            short_gap = tr.miss <= self.recover_max_gap
            not_over_recovered = tr.recovered_streak < self.recover_without_confirm_max
            if not (stable and short_gap and not_over_recovered):
                continue
            recovered_conf = max(float(conf), 0.25 * tr.conf)
            tr.update(box.astype(float), recovered_conf, recovered=True)
            used_tracks.add(mi)
            used_low.add(li)
            outputs.append((box, recovered_conf, "recovered", tr.tid))

        for i, tr in enumerate(self.tracks):
            if i not in used_tracks:
                tr.decay_miss()
        self.tracks = [tr for tr in self.tracks if tr.miss <= self.max_miss]
        return outputs, len(used_low), len(self.tracks)


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


def draw(frame, outputs):
    out = frame.copy()
    for box, conf, kind, tid in outputs:
        color = (0, 255, 0) if kind == "high" else (0, 165, 255)
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{kind[0].upper()} id{tid} {conf:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def run_video(model, args, recover: TrackRecover, idx: int, name: str, path: Path, out_dir: Path):
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

    vdir = out_dir / safe_name(Path(name).stem, idx)
    vdir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(str(vdir / f"{vdir.name}_trackrecover.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))

    rows = []
    frames = frames_out = total_out = recovered_frames = recovered_total = 0
    conf_accum = []
    raw = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cur_raw = raw
        raw += 1
        if cur_raw % stride:
            continue
        res = model.predict(frame, conf=args.low_conf, iou=args.iou, device=args.device, verbose=False)[0]
        if res.boxes is not None and len(res.boxes):
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
        else:
            boxes = np.zeros((0, 4))
            confs = np.zeros((0,))
        high_mask = confs >= args.high_conf
        low_mask = (confs >= args.low_conf) & (confs < args.high_conf)
        outputs, n_recovered, n_tracks = recover.step(boxes[high_mask], confs[high_mask], boxes[low_mask], confs[low_mask])

        frames += 1
        if outputs:
            frames_out += 1
            total_out += len(outputs)
            conf_accum.extend([o[1] for o in outputs])
        if n_recovered:
            recovered_frames += 1
            recovered_total += n_recovered
        rows.append([frames - 1, cur_raw, int(high_mask.sum()), int(low_mask.sum()), len(outputs), n_recovered, n_tracks])
        if writer is not None:
            writer.write(draw(frame, outputs))

    cap.release()
    if writer is not None:
        writer.release()
    with open(vdir / "frame_stats.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["kept_idx", "raw_frame", "high_det", "low_candidates", "final_det", "recovered", "active_tracks"])
        wr.writerows(rows)
    rec = {
        "index": idx,
        "video": name,
        "out_dir": vdir.name,
        "total_frames": total,
        "stride": stride,
        "frames_inferred": frames,
        "frames_with_det": frames_out,
        "total_det": total_out,
        "det_rate": round(frames_out / frames, 4) if frames else 0.0,
        "mean_conf": round(float(np.mean(conf_accum)), 4) if conf_accum else 0.0,
        "avg_det_per_frame": round(total_out / frames, 3) if frames else 0.0,
        "recovered_frames": recovered_frames,
        "recovered_total": recovered_total,
        "recover_rate": round(recovered_frames / frames, 4) if frames else 0.0,
    }
    with open(vdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return rec


def run_config(args, low_conf: float, match_iou: float, max_miss: int, tag: str):
    out_dir = Path(args.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    recover = TrackRecover(
        match_iou=match_iou,
        center_ratio=args.center_ratio,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        max_miss=max_miss,
        init_min_hits=args.init_min_hits,
        recover_max_gap=args.recover_max_gap,
        recover_without_confirm_max=args.recover_without_confirm_max,
    )
    old_low = args.low_conf
    args.low_conf = low_conf
    rows = []
    for idx, name, path in collect_videos(args.indices, args.source):
        rec = run_video(model, args, recover, idx, name, path, out_dir)
        rows.append(rec)
        print(f"[{tag}] #{idx} {name} -> det_rate={rec.get('det_rate')} recovered={rec.get('recovered_total')} mean_conf={rec.get('mean_conf')}", flush=True)
    args.low_conf = old_low
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return rows


def flist(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def ilist(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--source", default="")
    ap.add_argument("--indices", default="1,2,7,10,11")
    ap.add_argument("--out_dir", default=str(OUT_ROOT))
    ap.add_argument("--high_conf", type=float, default=0.25)
    ap.add_argument("--low_conf", default="0.05")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--match_iou", default="0.25")
    ap.add_argument("--center_ratio", type=float, default=0.8)
    ap.add_argument("--scale_min", type=float, default=0.5)
    ap.add_argument("--scale_max", type=float, default=2.0)
    ap.add_argument("--max_miss", default="3")
    ap.add_argument("--init_min_hits", type=int, default=2)
    ap.add_argument("--recover_max_gap", type=int, default=2)
    ap.add_argument("--recover_without_confirm_max", type=int, default=5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max_keep", type=int, default=500)
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()

    if not Path(args.weights).is_file():
        raise FileNotFoundError(args.weights)

    low_confs = flist(args.low_conf)
    match_ious = flist(args.match_iou)
    max_misses = ilist(args.max_miss)
    if not args.grid:
        low_confs, match_ious, max_misses = low_confs[:1], match_ious[:1], max_misses[:1]

    all_rows = []
    for low in low_confs:
        for miou in match_ious:
            for mm in max_misses:
                tag = f"low{low:.2f}_miou{miou:.2f}_miss{mm}".replace(".", "p")
                rows = run_config(args, low, miou, mm, tag)
                for r in rows:
                    all_rows.append({"low_conf": low, "match_iou": miou, "max_miss": mm, **r})

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "grid_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    keys = ["low_conf", "match_iou", "max_miss", "index", "video", "det_rate", "mean_conf", "avg_det_per_frame", "recovered_frames", "recovered_total", "recover_rate", "frames_inferred"]
    with open(out_root / "grid_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in all_rows:
            wr.writerow({k: r.get(k, "") for k in keys})
    print(f"\n[OK] 输出: {out_root}")


if __name__ == "__main__":
    main()
