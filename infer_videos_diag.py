"""用 v1(aux_p3main best.pt) 对测试视频做推理 + 过程变量诊断可视化。

每个视频输出到 result_p2_aux_p3main/video_test_v1/<vNN_...>/：
  - <name>_annotated.mp4   检测框+conf（conf<0.4 黄色高亮，便于找误检/弱响应）
  - perframe.csv           每帧 n_det / mean_conf / max_conf
  - diag_fXXXXXX.jpg       抽样帧诊断网格：原图+框 | P2(ResPAM@19)热图 | P3主检(22)热图
并在根目录写 summary.json / summary.csv 汇总所有视频统计。

中文路径鲁棒：读取失败时回退到临时 ASCII 副本；输出目录全用 ASCII（映射见 summary）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import torch

FINALMODEL = os.path.dirname(os.path.abspath(__file__))
ULOCAL = os.path.join(FINALMODEL, "ultralytics_local")
sys.path.insert(0, ULOCAL)
sys.path.insert(0, FINALMODEL)

import ultralytics_rainforest  # noqa: F401  注册 ResPAM 等自定义模块
from ultralytics import YOLO

WEIGHTS = os.path.join(FINALMODEL, "result_p2_aux_p3main", "fold_1", "weights", "best.pt")
VIDEO_DIR = r"D:\college\college3\monkey\用于测试的视频"
OUT_ROOT = os.path.join(FINALMODEL, "result_p2_aux_p3main", "video_test_v1")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
LOW_CONF = 0.4  # 低于此 conf 的框用黄色高亮（潜在误检/弱响应）


def safe_name(stem: str, idx: int) -> str:
    ascii_part = "".join(c for c in stem if c.isascii() and (c.isalnum() or c in "-_"))[:30]
    return f"v{idx:02d}_{ascii_part}" if ascii_part else f"v{idx:02d}"


def open_video(path: str):
    """打开视频，中文路径失败时回退到临时 ASCII 副本。返回 (cap, tmp_path_or_None)。"""
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap, None
    cap.release()
    tmp = os.path.join(tempfile.gettempdir(), f"_vid_{abs(hash(path)) % 10**8}{os.path.splitext(path)[1]}")
    shutil.copy(path, tmp)
    cap = cv2.VideoCapture(tmp)
    return cap, (tmp if cap.isOpened() else None)


def feat_to_heat(feat: torch.Tensor, w: int, h: int) -> np.ndarray:
    """[1,C,H,W] -> 通道均值归一化 -> JET 热力图 (h,w,3)。"""
    f = feat[0].float().mean(0)
    f = (f - f.min()) / (f.max() - f.min() + 1e-6)
    f = (f.cpu().numpy() * 255).astype("uint8")
    f = cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(f, cv2.COLORMAP_JET)


def draw_dets(frame: np.ndarray, boxes, confs) -> np.ndarray:
    out = frame.copy()
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        color = (0, 255, 0) if c >= LOW_CONF else (0, 215, 255)  # 绿 / 黄(BGR)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(out, f"{c:.2f}", (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max_frames", type=int, default=0, help="每视频最多处理帧数，0=全部（调试用小值）")
    ap.add_argument("--max_keep", type=int, default=1000,
                    help="自适应抽帧：每视频实际推理帧数上限；长视频自动按 stride 抽帧（标注视频降fps保持时长）")
    ap.add_argument("--n_diag", type=int, default=8, help="每视频抽样诊断帧数")
    ap.add_argument("--videos", type=str, default="", help="逗号分隔的文件名子串过滤；空=全部")
    ap.add_argument("--indices", type=str, default="", help="逗号分隔的1基序号过滤（按排序后顺序，避免中文命令行编码问题）")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--weights", type=str, default=WEIGHTS, help="权重路径，默认 v1(aux_p3main)")
    ap.add_argument("--out_subdir", type=str, default="video_test_v1", help="输出子目录名")
    ap.add_argument("--no_diag", action="store_true", help="跳过 P2/P3 激活热图（换模型对照时用，层不对应）")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)  # 实时进度（管道/重定向时也刷新）
    except Exception:
        pass

    weights = args.weights
    out_root = os.path.join(os.path.dirname(OUT_ROOT), args.out_subdir)
    assert os.path.isfile(weights), f"权重不存在: {weights}"
    assert os.path.isdir(VIDEO_DIR), f"视频目录不存在: {VIDEO_DIR}"
    os.makedirs(out_root, exist_ok=True)

    all_vids = sorted(f for f in os.listdir(VIDEO_DIR) if os.path.splitext(f)[1].lower() in VIDEO_EXTS)
    vid_items = list(enumerate(all_vids, start=1))  # 保留全局1基序号
    if args.indices:
        want = {int(x) for x in args.indices.split(",") if x.strip()}
        vid_items = [(i, v) for i, v in vid_items if i in want]
    if args.videos:
        subs = [s for s in args.videos.split(",") if s]
        vid_items = [(i, v) for i, v in vid_items if any(s in v for s in subs)]
    print(f"[INFO] 待处理视频 {len(vid_items)} 个，权重={weights}，输出={out_root}")

    model = YOLO(weights)
    net = model.model

    feats: dict[str, torch.Tensor] = {}
    if not args.no_diag:
        net.model[19].register_forward_hook(lambda m, i, o: feats.__setitem__("p2", o))
        net.model[22].register_forward_hook(lambda m, i, o: feats.__setitem__("p3", o))

    summary = []
    for idx, vname in vid_items:
        vpath = os.path.join(VIDEO_DIR, vname)
        sname = safe_name(os.path.splitext(vname)[0], idx)
        vout = os.path.join(out_root, sname)
        os.makedirs(vout, exist_ok=True)

        cap, tmp = open_video(vpath)
        if not cap.isOpened():
            print(f"[WARN] 无法打开: {vname}")
            summary.append({"index": idx, "video": vname, "error": "open_failed"})
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_proc = total if args.max_frames <= 0 else min(total, args.max_frames)
        stride = max(1, int(np.ceil(n_proc / args.max_keep))) if args.max_keep > 0 else 1
        n_keep = (n_proc + stride - 1) // stride
        diag_set = set(np.linspace(0, max(0, n_keep - 1), num=min(args.n_diag, max(1, n_keep)), dtype=int).tolist())

        out_fps = max(1.0, fps / stride)  # 抽帧后降fps保持时长一致
        writer = cv2.VideoWriter(os.path.join(vout, f"{sname}_annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
        rows = []
        frames_with_det = 0
        total_det = 0
        conf_accum = []
        fi = 0   # 已推理（保留）帧计数
        ri = 0   # 原始读取帧计数
        while True:
            ret, frame = cap.read()
            if not ret or (args.max_frames > 0 and ri >= args.max_frames):
                break
            cur_raw = ri
            ri += 1
            if cur_raw % stride != 0:
                continue
            res = model.predict(frame, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
            else:
                xyxy, confs = np.zeros((0, 4)), np.zeros((0,))
            n = len(confs)
            if n:
                frames_with_det += 1
                total_det += n
                conf_accum.extend(confs.tolist())
            writer.write(draw_dets(frame, xyxy, confs))
            rows.append([fi, cur_raw, n,
                         float(confs.mean()) if n else 0.0,
                         float(confs.max()) if n else 0.0])

            if fi in diag_set and "p2" in feats and "p3" in feats:
                base = draw_dets(frame, xyxy, confs)
                h2 = cv2.addWeighted(frame, 0.55, feat_to_heat(feats["p2"], W, H), 0.45, 0)
                h3 = cv2.addWeighted(frame, 0.55, feat_to_heat(feats["p3"], W, H), 0.45, 0)
                for im, tag in ((base, "det"), (h2, "P2-ResPAM@19"), (h3, "P3-main@22")):
                    cv2.putText(im, tag, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                grid = np.hstack([base, h2, h3])
                ok, buf = cv2.imencode(".jpg", grid)
                if ok:
                    buf.tofile(os.path.join(vout, f"diag_f{fi:06d}.jpg"))
            fi += 1

        cap.release()
        writer.release()
        if tmp and os.path.isfile(tmp):
            os.remove(tmp)

        with open(os.path.join(vout, "perframe.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kept_idx", "raw_frame", "n_det", "mean_conf", "max_conf"])
            w.writerows(rows)

        rec = {
            "index": idx, "video": vname, "out_dir": sname,
            "total_frames": total, "stride": stride, "frames_inferred": fi,
            "frames_with_det": frames_with_det, "total_det": total_det,
            "det_rate": round(frames_with_det / fi, 4) if fi else 0.0,
            "mean_conf": round(float(np.mean(conf_accum)), 4) if conf_accum else 0.0,
            "avg_det_per_frame": round(total_det / fi, 3) if fi else 0.0,
        }
        summary.append(rec)
        print(f"[{idx}/{len(all_vids)}] {vname} -> total={total} stride={stride} infer={fi} "
              f"det_rate={rec['det_rate']} mean_conf={rec['mean_conf']} avg_det/frame={rec['avg_det_per_frame']}")

    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    keys = ["index", "video", "out_dir", "total_frames", "stride", "frames_inferred",
            "frames_with_det", "total_det", "det_rate", "mean_conf", "avg_det_per_frame"]
    with open(os.path.join(out_root, "summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in summary:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\n[OK] 全部完成，输出根目录: {out_root}")


if __name__ == "__main__":
    main()
