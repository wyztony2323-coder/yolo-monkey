"""No-train P2 temporal memory inference for YOLOv8s-P2/ResPAM videos.

This is a first-stage architecture validation tool:
  - no YAML change
  - no training
  - no random conv/fuse parameters
  - only blends the P2 feature passed into Detect with an EMA memory

The wrapper patches the final Detect.forward at inference time and modifies only
Detect input x[0] (P2), leaving P3/P4/P5 and the PAN path unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

FINALMODEL = Path(__file__).resolve().parents[1]
REPO = FINALMODEL.parent
ULOCAL = FINALMODEL / "ultralytics_local"
sys.path.insert(0, str(ULOCAL))
sys.path.insert(0, str(FINALMODEL))

import ultralytics_rainforest  # noqa: F401  registers custom modules
from ultralytics import YOLO

VIDEO_DIR = Path(r"D:\college\college3\monkey\用于测试的视频")
DEFAULT_WEIGHTS = FINALMODEL / "result_p2_respam_p2only_ft800" / "fold_1" / "weights" / "best.pt"
OUT_ROOT = FINALMODEL / "result_temporal_p2"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class TemporalP2Controller:
    alpha: float = 0.15
    decay: float = 0.85
    enable_memory: bool = True

    def __post_init__(self):
        self.memory: torch.Tensor | None = None
        self.pending: torch.Tensor | None = None

    def reset_memory(self):
        self.memory = None
        self.pending = None

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Blend current P2 with previous EMA memory. No random parameters."""
        if not self.enable_memory or self.alpha <= 0:
            self.pending = x.detach()
            return x
        if self.memory is None or self.memory.shape != x.shape:
            self.pending = x.detach()
            return x
        mem = self.memory.to(device=x.device, dtype=x.dtype)
        self.pending = x.detach()
        return x + float(self.alpha) * (mem - x)

    def commit(self, allow_update: bool):
        """Update memory after detection results are known."""
        if self.pending is None:
            return
        x = self.pending.detach()
        if self.memory is None or self.memory.shape != x.shape:
            if allow_update:
                self.memory = x
            self.pending = None
            return
        mem = self.memory.to(device=x.device, dtype=x.dtype)
        if allow_update:
            self.memory = (float(self.decay) * mem + (1.0 - float(self.decay)) * x).detach()
        else:
            self.memory = (float(self.decay) * mem).detach()
        self.pending = None


def install_temporal_p2(model: YOLO, controller: TemporalP2Controller):
    """Patch final Detect.forward so only P2 detect input is temporally blended."""
    det = model.model.model[-1]
    orig_forward = det.forward

    def forward(self, x):
        if not self.training and isinstance(x, list) and len(x) >= 1:
            x = list(x)
            x[0] = controller.apply(x[0])
        return orig_forward(x)

    det.forward = types.MethodType(forward, det)
    return det


def safe_name(stem: str, idx: int) -> str:
    ascii_part = "".join(c for c in stem if c.isascii() and (c.isalnum() or c in "-_"))[:30]
    return f"v{idx:02d}_{ascii_part}" if ascii_part else f"v{idx:02d}"


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    return cap


def draw_dets(frame: np.ndarray, boxes, confs) -> np.ndarray:
    out = frame.copy()
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        color = (0, 255, 0) if c >= 0.4 else (0, 215, 255)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            out,
            f"{c:.2f}",
            (int(x1), max(0, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def collect_videos(indices: str = "", source: str = ""):
    if source:
        p = Path(source)
        return [(1, p.name, p)]
    all_vids = sorted(f for f in os.listdir(VIDEO_DIR) if Path(f).suffix.lower() in VIDEO_EXTS)
    items = list(enumerate(all_vids, start=1))
    if indices:
        want = {int(x) for x in indices.split(",") if x.strip()}
        items = [(i, v) for i, v in items if i in want]
    return [(i, v, VIDEO_DIR / v) for i, v in items]


def run_video(
    model: YOLO,
    controller: TemporalP2Controller,
    video_path: Path,
    video_name: str,
    global_idx: int,
    out_dir: Path,
    conf: float,
    iou: float,
    update_conf: float,
    device: str,
    max_keep: int,
    save_video: bool,
):
    controller.reset_memory()
    cap = open_video(video_path)
    if not cap.isOpened():
        return {"index": global_idx, "video": video_name, "error": "open_failed"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(np.ceil(total / max_keep))) if max_keep > 0 else 1
    out_fps = max(1.0, fps / stride)

    vout_dir = out_dir / safe_name(Path(video_name).stem, global_idx)
    vout_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if save_video:
        writer = cv2.VideoWriter(
            str(vout_dir / f"{vout_dir.name}_temporal.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            out_fps,
            (w, h),
        )

    rows = []
    frames = 0
    frames_with_det = 0
    total_det = 0
    conf_accum = []
    raw_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cur_raw = raw_idx
        raw_idx += 1
        if cur_raw % stride != 0:
            continue

        res = model.predict(frame, conf=conf, iou=iou, device=device, verbose=False)[0]
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            max_conf = float(confs.max())
        else:
            xyxy = np.zeros((0, 4))
            confs = np.zeros((0,))
            max_conf = 0.0

        allow_update = (update_conf <= 0) or (max_conf >= update_conf)
        controller.commit(allow_update)

        n = int(len(confs))
        frames += 1
        if n:
            frames_with_det += 1
            total_det += n
            conf_accum.extend(confs.tolist())
        rows.append([frames - 1, cur_raw, n, float(confs.mean()) if n else 0.0, max_conf, int(allow_update)])

        if writer is not None:
            writer.write(draw_dets(frame, xyxy, confs))

    cap.release()
    if writer is not None:
        writer.release()

    with open(vout_dir / "frame_stats.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["kept_idx", "raw_frame", "n_det", "mean_conf", "max_conf", "memory_update"])
        wr.writerows(rows)

    rec = {
        "index": global_idx,
        "video": video_name,
        "out_dir": vout_dir.name,
        "total_frames": total,
        "stride": stride,
        "frames_inferred": frames,
        "frames_with_det": frames_with_det,
        "total_det": total_det,
        "det_rate": round(frames_with_det / frames, 4) if frames else 0.0,
        "mean_conf": round(float(np.mean(conf_accum)), 4) if conf_accum else 0.0,
        "avg_det_per_frame": round(total_det / frames, 3) if frames else 0.0,
    }
    with open(vout_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return rec


def run_config(args, alpha: float, decay: float, update_conf: float, tag: str):
    out_dir = Path(args.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    controller = TemporalP2Controller(alpha=alpha, decay=decay, enable_memory=alpha > 0)
    model = YOLO(args.weights)
    install_temporal_p2(model, controller)

    summary = []
    for idx, name, path in collect_videos(args.indices, args.source):
        rec = run_video(
            model=model,
            controller=controller,
            video_path=path,
            video_name=name,
            global_idx=idx,
            out_dir=out_dir,
            conf=args.conf,
            iou=args.iou,
            update_conf=update_conf,
            device=args.device,
            max_keep=args.max_keep,
            save_video=args.save_video,
        )
        summary.append(rec)
        print(
            f"[{tag}] #{idx} {name} -> det_rate={rec.get('det_rate')} "
            f"mean_conf={rec.get('mean_conf')} avg_det/frame={rec.get('avg_det_per_frame')}",
            flush=True,
        )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        keys = [
            "index",
            "video",
            "total_frames",
            "stride",
            "frames_inferred",
            "frames_with_det",
            "total_det",
            "det_rate",
            "mean_conf",
            "avg_det_per_frame",
        ]
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in summary:
            wr.writerow({k: r.get(k, "") for k in keys})
    return summary


def parse_float_list(text: str):
    return [float(x) for x in text.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--source", default="")
    ap.add_argument("--indices", default="1,2,7,10,11")
    ap.add_argument("--out_dir", default=str(OUT_ROOT))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--alpha", default="0.15")
    ap.add_argument("--decay", default="0.85")
    ap.add_argument("--update_conf", default="0.25")
    ap.add_argument("--device", default="0")
    ap.add_argument("--max_keep", type=int, default=500)
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()

    if not Path(args.weights).is_file():
        raise FileNotFoundError(args.weights)

    alphas = parse_float_list(args.alpha)
    decays = parse_float_list(args.decay)
    updates = parse_float_list(args.update_conf)
    if not args.grid:
        alphas = alphas[:1]
        decays = decays[:1]
        updates = updates[:1]

    all_rows = []
    for alpha in alphas:
        for decay in decays:
            for update_conf in updates:
                tag = f"a{alpha:.2f}_d{decay:.2f}_u{update_conf:.2f}".replace(".", "p")
                rows = run_config(args, alpha, decay, update_conf, tag)
                for r in rows:
                    all_rows.append({"alpha": alpha, "decay": decay, "update_conf": update_conf, **r})

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "grid_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    with open(out_root / "grid_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        keys = [
            "alpha",
            "decay",
            "update_conf",
            "index",
            "video",
            "det_rate",
            "mean_conf",
            "avg_det_per_frame",
            "frames_inferred",
        ]
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in all_rows:
            wr.writerow({k: r.get(k, "") for k in keys})
    print(f"\n[OK] 输出: {out_root}")


if __name__ == "__main__":
    main()
