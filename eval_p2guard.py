"""Step1（无训练）：respam_p2only(4头) 的 head-specific 推理过滤验证。

利用本地 fork 的 Detect.postprocess 里 self.strides（per-anchor 来源尺度，P2=最小stride），
在 top-k 之前只对 P2 头 anchor 施加 (p2_conf + 面积上限)，对 P3/P4/P5 施加 main_conf。
对比 baseline(统一conf) vs p2guard 的逐视频检测率。
"""
from __future__ import annotations

import argparse
import os
import types

import cv2
import numpy as np
import torch

import infer_videos_diag as iv
from ultralytics import YOLO

RESPAM = os.path.join(iv.FINALMODEL, "result_p2_respam_p2only_ft800", "fold_1", "weights", "best.pt")


def install_p2guard(model, p2_conf: float, main_conf: float, p2_max_area: float):
    det = model.model.model[-1]  # Detect
    orig_post = det.__class__.postprocess  # 未绑定的原始实现

    def postprocess(self, preds):
        boxes, scores = preds.split([4, self.nc], dim=-1)        # boxes[B,A,4] xyxy(px), scores[B,A,nc]
        strides = self.strides.reshape(-1)                       # [A] per-anchor stride
        is_p2 = strides <= (strides.min() + 0.5)                 # P2 = 最小stride
        ih = int(self.shape[2]) * int(self.stride[0])
        iw = int(self.shape[3]) * int(self.stride[0])
        w = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
        h = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
        ar = (w * h) / float(ih * iw)                            # [B,A] 面积占比
        p2m = is_p2.view(1, -1, 1)
        kill_p2 = p2m & ((scores < p2_conf) | (ar.unsqueeze(-1) > p2_max_area))
        kill_main = (~p2m) & (scores < main_conf)
        scores = scores.masked_fill(kill_p2 | kill_main, 0.0)
        return orig_post(self, torch.cat([boxes, scores], dim=-1))

    det.postprocess = types.MethodType(postprocess, det)


def run(model, vpath, base_conf, max_keep):
    cap, tmp = iv.open_video(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(np.ceil(total / max_keep))) if max_keep > 0 else 1
    fwd = 0
    ndet = 0
    confs_all = []
    big = 0  # 大框(>0.05面积)数量，反映大目标误检倾向
    ri = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if ri % stride:
            ri += 1
            continue
        ri += 1
        res = model.predict(frame, conf=base_conf, iou=0.5, device="0", verbose=False)[0]
        fwd += 1
        if res.boxes is not None and len(res.boxes):
            cf = res.boxes.conf.cpu().numpy()
            xy = res.boxes.xyxy.cpu().numpy()
            H, W = frame.shape[:2]
            arr = ((xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])) / (W * H)
            ndet += 1
            confs_all.extend(cf.tolist())
            big += int((arr > 0.05).sum())
    cap.release()
    if tmp and os.path.isfile(tmp):
        os.remove(tmp)
    return {
        "frames": fwd, "stride": stride,
        "det_rate": round(ndet / fwd, 4) if fwd else 0,
        "mean_conf": round(float(np.mean(confs_all)), 4) if confs_all else 0,
        "big_boxes": big,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=str, default="7,1,2")
    ap.add_argument("--p2_conf", type=float, default=0.18)
    ap.add_argument("--main_conf", type=float, default=0.28)
    ap.add_argument("--p2_max_area", type=float, default=0.05)
    ap.add_argument("--base_conf", type=float, default=0.25)
    ap.add_argument("--max_keep", type=int, default=800)
    args = ap.parse_args()

    all_vids = sorted(f for f in os.listdir(iv.VIDEO_DIR)
                      if os.path.splitext(f)[1].lower() in iv.VIDEO_EXTS)
    want = [int(x) for x in args.indices.split(",") if x.strip()]
    sel = [(i, all_vids[i - 1]) for i in want]

    print(f"[INFO] baseline conf={args.base_conf} | guard p2={args.p2_conf} main={args.main_conf} area<={args.p2_max_area}")
    print(f"权重: {RESPAM}\n")

    model = YOLO(RESPAM)
    print("=== BASELINE (统一 conf) ===")
    base = {}
    for i, v in sel:
        base[i] = run(model, os.path.join(iv.VIDEO_DIR, v), args.base_conf, args.max_keep)
        print(f"  #{i} {v}: {base[i]}")

    install_p2guard(model, args.p2_conf, args.main_conf, args.p2_max_area)
    print("\n=== P2-GUARD (head-specific) ===")
    guard = {}
    for i, v in sel:
        guard[i] = run(model, os.path.join(iv.VIDEO_DIR, v), 0.05, args.max_keep)  # 低conf放行，由头阈值主导
        print(f"  #{i} {v}: {guard[i]}")

    print("\n=== 对比 (det_rate / mean_conf / big_boxes) ===")
    for i, v in sel:
        b, g = base[i], guard[i]
        print(f"  #{i} {v[:24]:24s} | base {b['det_rate']:.3f}/{b['mean_conf']:.2f}/{b['big_boxes']:4d}"
              f"  ->  guard {g['det_rate']:.3f}/{g['mean_conf']:.2f}/{g['big_boxes']:4d}")


if __name__ == "__main__":
    main()
