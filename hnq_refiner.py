"""Crop-based Hard-Negative-aware Quality (HNQ) refiner.

This module is deliberately independent from YOLO internals. It consumes
candidate crops and geometry vectors, then predicts:
  - gibbon logit
  - box quality logit
  - bbox delta
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


GEOM_DIM = 12


def cv_imread_unicode(path: str | Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def xyxy_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def nms_np(boxes: np.ndarray, scores: np.ndarray, iou_th: float = 0.65, topk: int = 300) -> list[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order) and len(keep) < topk:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        ious = xyxy_iou_np(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_th]
    return keep


def apply_delta_np(boxes: np.ndarray, deltas: np.ndarray, scale: float = 0.25) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    d = np.clip(deltas, -0.2, 0.2) * scale
    bw = np.maximum(1.0, boxes[:, 2] - boxes[:, 0])
    bh = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    bx = (boxes[:, 0] + boxes[:, 2]) / 2
    by = (boxes[:, 1] + boxes[:, 3]) / 2
    gx = bx + d[:, 0] * bw
    gy = by + d[:, 1] * bh
    gw = bw * np.exp(d[:, 2])
    gh = bh * np.exp(d[:, 3])
    return np.stack([gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2], axis=1).astype(np.float32)


class HNQDataset(Dataset):
    def __init__(self, csv_path: str | Path, split: str = "train", img_size: int = 96):
        self.csv_path = Path(csv_path)
        self.img_size = img_size
        with open(self.csv_path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r.get("split", "train") == split]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        im = cv_imread_unicode(r["crop_path"])
        if im is None:
            im = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        geom = torch.tensor([float(r[f"g{i}"]) for i in range(GEOM_DIM)], dtype=torch.float32)
        label = torch.tensor([float(r["label"])], dtype=torch.float32)
        q = torch.tensor([float(r["q_box"])], dtype=torch.float32)
        delta = torch.tensor([float(r[k]) for k in ("tx", "ty", "tw", "th")], dtype=torch.float32)
        return x, geom, label, q, delta


class HNQRefiner(nn.Module):
    def __init__(self, geom_dim: int = GEOM_DIM, width: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.geom = nn.Sequential(nn.Linear(geom_dim, width), nn.SiLU(inplace=True), nn.Linear(width, width))
        self.fuse = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU(inplace=True), nn.Dropout(0.1))
        self.cls = nn.Linear(width, 1)
        self.quality = nn.Linear(width, 1)
        self.delta = nn.Linear(width, 4)

    def forward(self, crop: torch.Tensor, geom: torch.Tensor):
        img_f = self.cnn(crop).flatten(1)
        geo_f = self.geom(geom)
        h = self.fuse(torch.cat([img_f, geo_f], dim=1))
        return self.cls(h), self.quality(h), self.delta(h)


def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    a = alpha * targets + (1 - alpha) * (1 - targets)
    return (a * (1 - pt).pow(gamma) * bce).mean()


def hnq_loss(outputs, label, q_target, delta_target, quality_w: float = 0.5, delta_w: float = 0.5):
    cls_logit, q_logit, delta = outputs
    label = label.float()
    cls_loss = focal_bce_with_logits(cls_logit, label)
    q_pred = torch.sigmoid(q_logit)
    quality_loss = F.smooth_l1_loss(q_pred, q_target.float())
    pos = label.view(-1) > 0.5
    if pos.any():
        delta_loss = F.smooth_l1_loss(torch.clamp(delta[pos], -0.5, 0.5), delta_target[pos].float())
    else:
        delta_loss = delta.sum() * 0.0
    total = cls_loss + quality_w * quality_loss + delta_w * delta_loss
    return total, {"cls": float(cls_loss.detach()), "quality": float(quality_loss.detach()), "delta": float(delta_loss.detach())}


def load_hnq(weights: str | Path, device: str = "cuda") -> HNQRefiner:
    ckpt = torch.load(weights, map_location=device)
    model = HNQRefiner()
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def iter_csv_rows(csv_path: str | Path) -> Iterable[dict]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)
