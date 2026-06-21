#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonkeyPlus / 自定义 YOLO 特征图可视化与诊断。

对 result_monkeyplus_p2 等权重，在单张图或视频帧上：
  - 抓取 MSCB / PAM / 各尺度 neck / Detect 前特征
  - 输出通道热力图、Top-K 通道栅格、尺度间激活对比
  - 生成 diagnose.json 便于定位「P2 弱 / MSCB 抑制 / PAM 过强」等问题

用法（在 finalmodel 目录）:
  python tools/visualize_monkeyplus_featuremaps.py \\
    --weights ../result_p2_only/fold_1/weights/best.pt \\
    --model_type yolov8s_p2_only \\
    --source ../海南长臂猿-宣传片.mp4 --frame 500

  # PowerShell：行末不要用 # 注释，否则 # 后内容会被当成额外参数
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).resolve().parent
_FINALMODEL = _SCRIPT_DIR.parent
_REPO_ROOT = _FINALMODEL.parent
_ULTRALYTICS_LOCAL = _FINALMODEL / "ultralytics_local"
# 顺序要点：ultralytics_local 与 finalmodel 必须排在仓库根目录之前，
# 否则会导入根目录旧版 ultralytics_rainforest（缺 ResPAM/SGF/HFRes/BiFPNLite），
# 导致加载新结构 best.pt 时反序列化失败。
for p in (_REPO_ROOT, _FINALMODEL, _ULTRALYTICS_LOCAL):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

import ultralytics_rainforest  # noqa: F401,E402
from ultralytics import YOLO  # noqa: E402

# 与 yolov8s-monkeyplus-p2.yaml 对齐的关键层（已通过 best.pt 枚举校验）
MONKEYPLUS_KEY_LAYERS = {
    2: "backbone_P2_C2f",
    4: "backbone_P3_C2f",
    6: "backbone_P4_C2f",
    7: "backbone_MSCB",
    9: "backbone_P5_C2f",
    10: "backbone_SPPF",
    13: "neck_P4_C2f",
    14: "neck_P4_PAM",
    17: "neck_P3_C2f",
    18: "neck_P3_PAM",
    21: "neck_P2_C2f",
    22: "neck_P2_PAM",
    26: "neck_P3_out_PAM",
    30: "neck_P4_out_PAM",
    34: "neck_P5_out_PAM",
    35: "Detect",
}

SCALE_LAYERS = {
    "P2": 22,
    "P3": 18,
    "P4": 14,
    "P5": 34,
}

# yolov8s-p2-only.yaml（已通过 best.pt 枚举；Detect 输入 18/21/24/27）
P2_ONLY_KEY_LAYERS = {
    2: "backbone_P2_C2f",
    4: "backbone_P3_C2f",
    6: "backbone_P4_C2f",
    8: "backbone_P5_C2f",
    9: "backbone_SPPF",
    12: "neck_P4_merge_C2f",
    15: "neck_P3_merge_C2f",
    18: "neck_P2_C2f",
    21: "neck_P3_C2f",
    24: "neck_P4_C2f",
    27: "neck_P5_C2f",
    28: "Detect",
}

P2_RESPAM_KEY_LAYERS = {
    **{k: v for k, v in P2_ONLY_KEY_LAYERS.items()},
    19: "neck_P2_ResPAM",
}

# yolov8s-p2-respam-sgf.yaml（含 HFRes/SGF 时层号后移；Detect 输入 31/22/25/28）
# 同样适用 wide 版（层号一致，仅通道不同）
P2_RESPAM_SGF_KEY_LAYERS = {
    2: "backbone_P2_C2f",
    4: "backbone_P3_C2f",
    6: "backbone_P4_C2f",
    8: "backbone_P5_C2f",
    9: "backbone_SPPF",
    12: "neck_P4_td_C2f",
    15: "neck_P3_td_C2f",
    18: "neck_P2_C2f",
    19: "neck_P2_ResPAM",
    22: "neck_P3_C2f",
    25: "neck_P4_C2f",
    28: "neck_P5_C2f",
    31: "neck_P2_SGF",
    32: "Detect",
}

MODEL_TYPE_CHOICES = [
    "auto",
    "yolov8s_monkeyplus_p2",
    "yolov8s_monkeyplus_p2_respam",
    "yolov8s_p2_only",
    "yolov8s_p2_only_cm30",
    "yolov8s_p2_only_cm30_ft800",
    "yolov8s_p2_wide",
    "yolov8s_p2_wide_respam",
    "yolov8s_p2_respam_p2only",
    "yolov8s_p2_respam_hfres",
    "yolov8s_p2_respam_sgf",
    "yolov8s_p2_wide_respam_sgf",
    "yolov8s_p2_bifpn_lite",
    "yolov8s",
]


def load_bgr(path: Path, frame_idx: int = 0) -> np.ndarray:
    suf = path.suffix.lower()
    if suf in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {path}")
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"无法读取视频帧 {frame_idx}: {path}")
        return frame
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"无法读取图像: {path}")
    return img


def letterbox(im: np.ndarray, new_shape: int = 640, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_w, pad_h = new_shape - nw, new_shape - nh
    pad_w, pad_h = pad_w // 2, pad_h // 2
    if (h, w) != (nh, nw):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=color)
    return im, r, (pad_w, pad_h)


def tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    x = t.detach().float()
    if x.ndim == 4:
        x = x[0]
    abs_x = x.abs()
    flat = abs_x.reshape(abs_x.shape[0], -1)
    ch_mean = flat.mean(dim=1)
    dead = (ch_mean < 1e-4).float().mean().item()
    return {
        "channels": int(x.shape[0]),
        "h": int(x.shape[1]),
        "w": int(x.shape[2]),
        "mean_abs": float(abs_x.mean().item()),
        "max_abs": float(abs_x.max().item()),
        "std": float(x.std().item()),
        "dead_channel_ratio": dead,
        "spatial_peak": float(abs_x.amax(dim=0).mean().item()),
    }


def activation_heatmap(t: torch.Tensor, up_to: Tuple[int, int]) -> np.ndarray:
    """通道均值绝对值激活图，上采样到 up_to (H,W)。"""
    x = t.detach().float()
    if x.ndim == 4:
        x = x[0]
    hm = x.abs().mean(dim=0).cpu().numpy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm = cv2.resize(hm, (up_to[1], up_to[0]), interpolation=cv2.INTER_LINEAR)
    return hm


def overlay_heatmap(bgr: np.ndarray, hm: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1 - alpha, color, alpha, 0)


def save_channel_grid(t: torch.Tensor, out_path: Path, n: int = 16) -> None:
    x = t.detach().float()
    if x.ndim == 4:
        x = x[0]
    c = x.shape[0]
    n = min(n, c)
    ch_mean = x.abs().reshape(c, -1).mean(dim=1)
    top_idx = torch.topk(ch_mean, n).indices.cpu().numpy()
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        ch = x[top_idx[i]].cpu().numpy()
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        ax.imshow(ch, cmap="viridis")
        ax.set_title(f"ch{top_idx[i]}", fontsize=7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


class FeatureHook:
    def __init__(self) -> None:
        self.cache: Dict[int, torch.Tensor] = {}
        self._handles: List[Any] = []

    def register(self, model: nn.Module, layer_indices: Dict[int, str]) -> None:
        layers = list(model.model)
        for idx, name in layer_indices.items():
            if idx >= len(layers):
                continue
            layer = layers[idx]

            def _hook(module, inp, out, layer_idx=idx):
                if isinstance(out, (list, tuple)):
                    o = out[0] if out else None
                else:
                    o = out
                if isinstance(o, torch.Tensor):
                    self.cache[layer_idx] = o.detach().cpu()

            self._handles.append(layer.register_forward_hook(_hook))

    def clear(self) -> None:
        self.cache.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _model_has_mscb(model: nn.Module) -> bool:
    return any(type(m).__name__ == "MSCB" for m in model.model)


def discover_key_layers(model: nn.Module) -> Dict[int, str]:
    if _model_has_mscb(model):
        name_map = MONKEYPLUS_KEY_LAYERS
    elif any(type(m).__name__ == "SGF" for m in model.model):
        name_map = P2_RESPAM_SGF_KEY_LAYERS
    elif any(type(m).__name__ == "ResPAM" for m in model.model):
        name_map = P2_RESPAM_KEY_LAYERS
    else:
        name_map = P2_ONLY_KEY_LAYERS
    out: Dict[int, str] = {}
    for i, m in enumerate(model.model):
        tname = type(m).__name__
        label = name_map.get(i)
        if label is not None:
            # 仅当层类型与标签一致时才用友好名（避免 p2-only 的 layer19=Conv 被标成 ResPAM）
            if "ResPAM" in label and tname != "ResPAM":
                label = None
            elif "SGF" in label and tname != "SGF":
                label = None
            elif "PAM" in label and "ResPAM" not in label and tname != "PAM":
                label = None
            elif "MSCB" in label and tname != "MSCB":
                label = None
        if label:
            out[i] = label
        elif tname in ("MSCB", "PAM", "ResPAM", "SGF", "BiFPNLite", "Detect"):
            out[i] = f"{tname}_{i}"
        elif tname == "C2f":
            out[i] = f"C2f_{i}"
        elif tname == "SPPF":
            out[i] = "SPPF"
    return dict(sorted(out.items()))


def infer_diagnosis_profile(model_type: str, stats: Dict[str, Dict[str, float]]) -> str:
    if model_type != "auto":
        if "monkeyplus" in model_type or model_type == "yolov8s":
            return "monkeyplus" if model_type != "yolov8s" else "baseline"
        return "p2_only"
    if any("MSCB" in k for k in stats):
        return "monkeyplus"
    if any("PAM_" in k for k in stats):
        return "monkeyplus"
    return "p2_only"


def run_diagnosis(stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """根据各层统计量给出可读诊断提示。"""
    hints: List[str] = []
    issues: List[str] = []

    def get_exact(*keys: str) -> Optional[Dict[str, float]]:
        for key in keys:
            if key in stats:
                return stats[key]
        for k, v in stats.items():
            if any(key in k for key in keys):
                return v
        return None

    p4_c2f = get_exact("backbone_P4_C2f")
    mscb = get_exact("MSCB_7", "MSCB")
    if p4_c2f and mscb:
        ratio = mscb["mean_abs"] / (p4_c2f["mean_abs"] + 1e-8)
        if ratio < 0.65:
            issues.append(
                f"MSCB 将 P4 特征均值从 {p4_c2f['mean_abs']:.3f} 压到 {mscb['mean_abs']:.3f}（×{ratio:.2f}），"
                "门控+随机初始化 MSCB 可能在 backbone 中段削弱语义（与预训练仅 56% 匹配一致）。"
            )

    pam_pairs = [
        ("P4", "neck_P4_C2f", "PAM_14"),
        ("P3", "neck_P3_C2f", "PAM_18"),
        ("P2", "neck_P2_C2f", "PAM_22"),
        ("P3_out", "neck_P3_out", "PAM_26"),
        ("P4_out", "neck_P4_out", "PAM_30"),
        ("P5_out", "neck_P5_out", "PAM_34"),
    ]
    scale_pam_after: Dict[str, float] = {}
    pam_drop_count = 0
    for sk, before_k, after_k in pam_pairs:
        bef = get_exact(before_k)
        aft = get_exact(after_k)
        if not bef or not aft:
            continue
        r = aft["mean_abs"] / (bef["mean_abs"] + 1e-8)
        scale_pam_after[sk] = aft["mean_abs"]
        if r < 0.6:
            pam_drop_count += 1
            issues.append(f"{sk}: PAM 后 mean_abs 降至 C2f 前的 {r:.0%}（{bef['mean_abs']:.3f}→{aft['mean_abs']:.3f}）。")

    if pam_drop_count >= 3:
        hints.append(
            "多个尺度 PAM 均明显压低激活幅度；若热力图未聚焦在猿体，可考虑 P2-only ResPAM 或减小 PAM 路权重。"
        )

    if len(scale_pam_after) >= 2:
        p2 = scale_pam_after.get("P2", scale_pam_after.get("P2_out", 0))
        p5 = scale_pam_after.get("P5_out", 0)
        if p2 and p5:
            p2_p5 = p2 / (p5 + 1e-8)
            if p2_p5 < 0.5:
                issues.append(f"P2 PAM 后激活仅为 P5 输出的 {p2_p5:.0%}，小目标头相对偏弱。")
            elif 0.85 <= p2_p5 <= 1.15:
                hints.append(f"P2/P5 PAM 后激活同量级（比≈{p2_p5:.2f}），P2 未塌缩；漏检更可能来自预训练/定位而非 P2 无响应。")

    for k, v in stats.items():
        if v["dead_channel_ratio"] > 0.35:
            issues.append(f"{k}: 死通道比例 {v['dead_channel_ratio']:.1%} 偏高。")

    return {
        "profile": "monkeyplus",
        "scale_pam_mean_abs": scale_pam_after,
        "hints": hints,
        "issues": issues,
        "summary": (
            "未发现明显异常，请结合叠加图人工查看小目标区域。"
            if not issues
            else "；".join(issues)
        ),
    }


def run_diagnosis_p2_only(stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """P2-only / P2+ResPAM：无 MSCB、无多路 PAM，对比各尺度 neck C2f。"""
    hints: List[str] = []
    issues: List[str] = []

    def get_exact(*keys: str) -> Optional[Dict[str, float]]:
        for key in keys:
            if key in stats:
                return stats[key]
        for k, v in stats.items():
            if any(key in k for key in keys):
                return v
        return None

    scale_c2f: Dict[str, float] = {}
    for sk, key in (
        ("P2", "neck_P2_C2f"),
        ("P3", "neck_P3_C2f"),
        ("P4", "neck_P4_C2f"),
        ("P5", "neck_P5_C2f"),
    ):
        v = get_exact(key)
        if v:
            scale_c2f[sk] = v["mean_abs"]

    p2_c2f = get_exact("neck_P2_C2f")
    respam = get_exact("neck_P2_ResPAM", "ResPAM_19")
    if p2_c2f and respam:
        r = respam["mean_abs"] / (p2_c2f["mean_abs"] + 1e-8)
        if r < 0.6:
            issues.append(
                f"P2 ResPAM 后 mean_abs 为 C2f 前的 {r:.0%}（{p2_c2f['mean_abs']:.3f}→{respam['mean_abs']:.3f}），"
                "残差注意力仍明显压低（检查 alpha 是否已学太大）。"
            )
        elif 0.9 <= r <= 1.1:
            hints.append(f"P2 ResPAM 后激活与 C2f 接近（×{r:.2f}），初值恒等策略有效。")
    elif p2_c2f and not respam:
        hints.append("纯 P2-only：无 ResPAM，neck 特征未被并联 PAM 压低，可与 MonkeyPlus 特征图对比。")

    if "P2" in scale_c2f and "P5" in scale_c2f:
        ratio = scale_c2f["P2"] / (scale_c2f["P5"] + 1e-8)
        if ratio < 0.35:
            issues.append(f"neck P2 C2f 激活仅为 P5 的 {ratio:.0%}，小目标分支偏弱。")
        elif ratio >= 0.5:
            hints.append(f"neck P2/P5 C2f 激活比≈{ratio:.2f}，P2 分支有响应（请对照热力图是否对准目标）。")

    if any("MSCB" in k for k in stats) or sum(1 for k in stats if "PAM_" in k) >= 2:
        issues.append("权重中仍含 MSCB/多路 PAM，与 yolov8s_p2_only 预期不符，请确认 --weights 路径。")

    return {
        "profile": "p2_only",
        "scale_neck_c2f_mean_abs": scale_c2f,
        "hints": hints,
        "issues": issues,
        "summary": (
            "P2-only 结构：未检测到 MSCB/多路 PAM 式幅值腰斩；请结合 18_neck_P2_C2f_heatmap 人工核对小目标区域。"
            if not issues
            else "；".join(issues)
        ),
    }


def run_diagnosis_for_model(model_type: str, stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    profile = infer_diagnosis_profile(model_type, stats)
    if profile == "p2_only":
        return run_diagnosis_p2_only(stats)
    return run_diagnosis(stats)


def plot_scale_comparison(
    stats: Dict[str, Dict[str, float]], out_path: Path, profile: str = "monkeyplus"
) -> None:
    labels, vals = [], []
    if profile == "p2_only":
        for sk in ("P2", "P3", "P4", "P5"):
            for k, v in stats.items():
                if sk in k and "neck" in k and "C2f" in k:
                    labels.append(k.replace("neck_", ""))
                    vals.append(v["mean_abs"])
                    break
        title = "各尺度 neck C2f 平均激活（P2-only）"
    else:
        for sk in ("P2", "P3", "P4", "P5"):
            for k, v in stats.items():
                if sk in k and "PAM" in k:
                    labels.append(k.replace("neck_", ""))
                    vals.append(v["mean_abs"])
                    break
        title = "各尺度 PAM 后平均激活强度"
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(labels)), vals, color=["#e74c3c", "#e67e22", "#3498db", "#2ecc71"][: len(labels)])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("mean |activation|")
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="MonkeyPlus 特征图可视化诊断")
    ap.add_argument(
        "--weights",
        type=str,
        default=str(_REPO_ROOT / "result_monkeyplus_p2" / "fold_1_phase2" / "weights" / "best.pt"),
    )
    ap.add_argument(
        "--source",
        type=str,
        default=str(_REPO_ROOT / "海南长臂猿-宣传片.mp4"),
        help="图像或视频路径",
    )
    ap.add_argument("--frame", type=int, default=0, help="视频帧索引")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(_REPO_ROOT / "result_monkeyplus_p2" / "feature_viz"),
    )
    ap.add_argument("--topk", type=int, default=16, help="通道栅格显示数")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument(
        "--model_type",
        type=str,
        choices=MODEL_TYPE_CHOICES,
        default="auto",
        help="诊断规则：auto=按权重是否含 MSCB/PAM 推断；p2_only 系列用 neck C2f 对比",
    )
    args = ap.parse_args()

    weights = Path(args.weights).resolve()
    source = Path(args.source).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = load_bgr(source, args.frame)
    cv2.imwrite(str(out_dir / "input_frame.jpg"), bgr)
    lb, _, _ = letterbox(bgr, args.imgsz)

    print(f"[Load] {weights}")
    yolo = YOLO(str(weights))
    model = yolo.model
    model.eval()

    key_layers = discover_key_layers(model)
    hook = FeatureHook()
    hook.register(model, key_layers)

    with torch.no_grad():
        pred = yolo.predict(
            lb,
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )[0]
    pred_path = out_dir / "prediction.jpg"
    cv2.imwrite(str(pred_path), pred.plot())

    hook.remove()

    h, w = lb.shape[:2]
    layer_stats: Dict[str, Dict[str, float]] = {}
    for idx in sorted(hook.cache.keys()):
        t = hook.cache[idx]
        label = key_layers.get(idx, f"layer_{idx}")
        layer_stats[label] = tensor_stats(t)

        hm = activation_heatmap(t, (h, w))
        cv2.imwrite(str(out_dir / f"{idx:02d}_{label}_heatmap.jpg"), overlay_heatmap(lb, hm))
        save_channel_grid(t, out_dir / f"{idx:02d}_{label}_channels.png", n=args.topk)

    profile = infer_diagnosis_profile(args.model_type, layer_stats)
    diagnose = run_diagnosis_for_model(args.model_type, layer_stats)
    diagnose["model_type"] = args.model_type
    diagnose["weights"] = str(weights)
    diagnose["source"] = str(source)
    diagnose["frame"] = args.frame
    diagnose["layer_stats"] = layer_stats
    diagnose["key_layers"] = {str(k): v for k, v in key_layers.items()}

    with open(out_dir / "diagnose.json", "w", encoding="utf-8") as f:
        json.dump(diagnose, f, ensure_ascii=False, indent=2)

    scale_png = out_dir / ("scale_neck_c2f_mean_abs.png" if profile == "p2_only" else "scale_pam_mean_abs.png")
    plot_scale_comparison(layer_stats, scale_png, profile=profile)

    title = "P2-only 特征图诊断" if profile == "p2_only" else "MonkeyPlus 特征图诊断"
    lines = [
        f"# {title}",
        f"- model_type: `{args.model_type}` (profile=`{profile}`)",
        f"- 权重: `{weights}`",
        f"- 输入: `{source}` (frame={args.frame})",
        f"- 输出目录: `{out_dir}`",
        "",
        "## 自动结论",
        diagnose["summary"],
        "",
        "## 问题项",
    ]
    lines.extend(f"- {x}" for x in diagnose.get("issues", [])) or ["- （无）"]
    lines.append("\n## 提示项")
    lines.extend(f"- {x}" for x in diagnose.get("hints", [])) or ["- （无）"]
    lines.append("\n## 各层 mean_abs（PAM / 关键层）")
    for k, v in sorted(layer_stats.items()):
        if any(x in k for x in ("PAM", "MSCB", "SPPF", "P2", "Detect")):
            lines.append(
                f"- {k}: mean_abs={v['mean_abs']:.4f}, dead_ch={v['dead_channel_ratio']:.2%}, "
                f"size={v['h']}x{v['w']}"
            )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n✅ 特征图已保存: {out_dir}")
    print(f"   预测叠加: {pred_path}")
    print(f"   诊断 JSON: {out_dir / 'diagnose.json'}")
    print(f"   报告: {out_dir / 'REPORT.md'}")
    if diagnose.get("issues"):
        print("\n⚠️  自动诊断问题:")
        for x in diagnose["issues"]:
            print(f"   - {x}")


if __name__ == "__main__":
    main()
