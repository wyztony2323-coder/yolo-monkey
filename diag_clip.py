"""针对指定视频的某个时间窗逐帧诊断（用于分析特定误检/漏检）。

复用 infer_videos_diag 的 best.pt 加载、hook、热图、绘框逻辑。
输出每帧: 原图+框(类别+conf) | P2(ResPAM@19) | P3(main@22) 热图网格。
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch

import infer_videos_diag as iv
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True, help="视频在排序列表中的1基序号")
    ap.add_argument("--start", type=float, required=True, help="起始秒")
    ap.add_argument("--end", type=float, required=True, help="结束秒")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--tag", type=str, default="clip")
    args = ap.parse_args()

    all_vids = sorted(f for f in os.listdir(iv.VIDEO_DIR)
                      if os.path.splitext(f)[1].lower() in iv.VIDEO_EXTS)
    vname = all_vids[args.index - 1]
    vpath = os.path.join(iv.VIDEO_DIR, vname)
    print(f"[INFO] 视频 #{args.index}: {vname}")

    out_dir = os.path.join(iv.FINALMODEL, "result_p2_aux_p3main", f"clip_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    model = YOLO(iv.WEIGHTS)
    net = model.model
    feats: dict[str, torch.Tensor] = {}
    net.model[19].register_forward_hook(lambda m, i, o: feats.__setitem__("p2", o))
    net.model[22].register_forward_hook(lambda m, i, o: feats.__setitem__("p3", o))

    cap, tmp = iv.open_video(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f0 = int(args.start * fps)
    f1 = int(args.end * fps)
    print(f"[INFO] fps={fps} 帧范围 {f0}..{f1} ({W}x{H})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    fi = f0
    while fi <= f1:
        ret, frame = cap.read()
        if not ret:
            break
        res = model.predict(frame, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
        names = res.names
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
        else:
            xyxy, confs, clss = np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), int)

        base = frame.copy()
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            color = (0, 255, 0) if c >= iv.LOW_CONF else (0, 215, 255)
            cv2.rectangle(base, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(base, f"{names.get(k, k)} {c:.2f}", (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        tsec = fi / fps
        cv2.putText(base, f"f{fi} t={tsec:.2f}s ndet={len(confs)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        if "p2" in feats and "p3" in feats:
            h2 = cv2.addWeighted(frame, 0.55, iv.feat_to_heat(feats["p2"], W, H), 0.45, 0)
            h3 = cv2.addWeighted(frame, 0.55, iv.feat_to_heat(feats["p3"], W, H), 0.45, 0)
            cv2.putText(h2, "P2-ResPAM@19", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(h3, "P3-main@22", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            grid = np.hstack([base, h2, h3])
        else:
            grid = base

        ok, buf = cv2.imencode(".jpg", grid)
        if ok:
            buf.tofile(os.path.join(out_dir, f"f{fi:06d}.jpg"))
        if len(confs):
            print(f"  f{fi} t={tsec:.2f}s -> " +
                  ", ".join(f"{names.get(k,k)}({c:.2f})" for c, k in zip(confs, clss)))
        fi += 1

    cap.release()
    if tmp and os.path.isfile(tmp):
        os.remove(tmp)
    print(f"[OK] 输出: {out_dir}")


if __name__ == "__main__":
    main()
