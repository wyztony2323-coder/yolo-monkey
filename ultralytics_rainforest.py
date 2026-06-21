"""
ultralytics_rainforest.py — Rainforest-YOLO paper-faithful module pack
=====================================================================

用途
----
为论文复现工程提供自定义模块与 Ultralytics 注册逻辑，核心模块包括：
  1. C2f_AdaConv   （YOLOv8 论文复现主路径）
  2. MSCB          （Multi-Scale Convolutional Block）
  3. PAM / ResPAM  （Parallel Attention；ResPAM 残差初值≈恒等）
  4. RST           （Residual Swin Transformer）

兼容性说明
--------
- 本文件保留了部分工程兼容实现，例如：
  - C3k2_AdaConv：用于兼容 YOLO11 / C3k2 风格结构
  - ST          ：用于消融或兼容旧版权重
- 论文主路径（YOLOv8s / Rainforest-YOLO）实际只需要：
  - C2f_AdaConv
  - MSCB
  - PAM / ResPAM
  - RST

设计原则
--------
- 优先保证训练/加载兼容性，不做会改变行为的“美化式重构”
- 保留 parse_model patch、多路径注册、旧权重兼容别名
- 对外建议把本文件视为“模块库”，而不是论文说明文档

使用方式
--------
训练脚本可直接：
    import ultralytics_rainforest   # 导入即注册
"""

from __future__ import annotations

import math
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics.nn import tasks
    from ultralytics.nn.modules.conv import Conv, DWConv
    from ultralytics.nn.modules.head import Detect
except ImportError:
    tasks = None
    Conv = None
    DWConv = None
    Detect = None

# ─── Conv fallback（无 Ultralytics 时）────────────────────────────────────────
if Conv is None:
    class Conv(nn.Module):  # type: ignore[no-redef]
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
            super().__init__()
            p = p if p is not None else (k - 1) // 2
            self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
            self.bn   = nn.BatchNorm2d(c2)
            self.act  = nn.SiLU() if act else nn.Identity()
        def forward(self, x):
            return self.act(self.bn(self.conv(x)))


def _ch(x: object) -> int:
    """通道数安全转 int（处理 list / tuple / int）。"""
    if isinstance(x, (list, tuple)):
        return sum(int(v) for v in x)
    return int(x)  # type: ignore[arg-type]


# =============================================================================
# 1. AdaConv family
#    论文主路径：C2f_AdaConv（YOLOv8）
#    兼容路径：C3k2_AdaConv（YOLO11 / C3k2 风格）
#    AdaConv = 两层 Conv-BN-SiLU + Context-Gate（GAP→MLP→Sigmoid 缩放输出）
# =============================================================================

class AdaConvBlock(nn.Module):
    """
    论文图4 Bottleneck_AdaConv：
      cv1(1×1-BN-SiLU) → cv2(3×3-BN-SiLU) → gate(GAP→MLP→Sigmoid) → 可选残差

    gate bias 初始化为 +3.0：sigmoid(3)≈0.95，使初始 gate≈1，
    避免输出≈0.5x、特征衰减半、梯度震荡。
    """
    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = max(1, int(c2 * e))
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and (c1 == c2)
        mid = max(c2 // 16, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c2, mid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c2, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[-2].bias, 3.0)  # FIX: gate 初始接近 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        g   = self.gate(out).view(out.shape[0], out.shape[1], 1, 1)
        out = out * g
        return (x + out) if self.add else out


class C3k2_AdaConv(nn.Module):
    """
    YOLO11 C3k2 结构，内部 Bottleneck 替换为 AdaConvBlock。
    YAML 用法：[-1, 1, C3k2_AdaConv, [c2, shortcut, e, n_bottlenecks]]
    parse_model patch 会在 args 头部插入 c1=ch[from]。
    """
    def __init__(self, c1: int, c2: int, shortcut: bool = False,
                 e: float = 0.25, n: int = 1):
        super().__init__()
        c1, c2 = _ch(c1), _ch(c2)
        self.c   = c_ = max(1, int(c2 * e))
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1)
        self.m   = nn.ModuleList(
            AdaConvBlock(c_, c_, shortcut=shortcut, e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

    def extra_repr(self) -> str:
        return f"c={self.c}, bottlenecks={len(self.m)}"


class C2f_AdaConv(nn.Module):
    """
    YOLOv8 C2f 结构，内部 Bottleneck 替换为 AdaConvBlock（e=0.5，split 而非 chunk）。
    保留用于 YOLOv8 消融对比；YOLO11s 版主体使用 C3k2_AdaConv。
    """
    def __init__(self, c1: int, c2: int, n: int = 1,
                 shortcut: bool = False, e: float = 0.5):
        super().__init__()
        c1, c2 = _ch(c1), _ch(c2)
        self.c   = c_ = max(1, int(c2 * e))
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1)
        self.m   = nn.ModuleList(
            AdaConvBlock(c_, c_, shortcut=shortcut) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


# =============================================================================
# 2. MSCB（Multi-Scale Convolutional Block）
#    论文 Section 3.2 / 图5：
#    三个并行非对称分支 (5×1,1×5) / (11×1,1×11) / (15×1,1×15)
#    每分支含 BN+ReLU（FIX-8）→ Cat → Conv1×1-Sigmoid（gate）→ × shortcut
#
# FIX-9：bc 使用 ceiling 除法 (c1+2)//3，避免 floor 除法损失最多 2 通道
# =============================================================================

class MSCB(nn.Module):
    """
    论文图5 Multi-Scale Convolutional Block：
      output = sigmoid(proj(cat(b1,b2,b3))) * shortcut(x)

    每分支：Conv(k×1, c1→bc) + BN + ReLU → Conv(1×k, bc→bc, dw) + BN
    gate bias 初始化为 +3.0，避免初期 gate≈0.5 导致特征衰减。
    """
    def __init__(self, c1: int, c2: int | None = None):
        super().__init__()
        c1 = _ch(c1)
        c2 = _ch(c2) if c2 is not None else c1
        # FIX-9: ceiling 除法，3 个分支通道之和 >= c1（最多多 2 通道，无信息丢失）
        bc = max((c1 + 2) // 3, 1)

        def _branch(k: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c1, bc, (k, 1), padding=(k // 2, 0), bias=False),
                nn.BatchNorm2d(bc),           # FIX-8
                nn.ReLU(inplace=True),         # FIX-8
                nn.Conv2d(bc, bc, (1, k), padding=(0, k // 2), groups=bc, bias=False),
                nn.BatchNorm2d(bc),            # FIX-8
            )

        self.b1 = _branch(5)
        self.b2 = _branch(11)
        self.b3 = _branch(15)

        self.proj     = nn.Conv2d(3 * bc, c2, 1, bias=True)
        self.gate_act = nn.Sigmoid()
        nn.init.constant_(self.proj.bias, 3.0)   # FIX: gate 初始接近 1

        self.shortcut = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        gate = self.gate_act(self.proj(feat))    # (0,1) gate
        return gate * self.shortcut(x)           # 论文图5 乘法符号


# =============================================================================
# 3. PAM（Parallel Attention Module）
#    论文 Section 3.3 / 图3 / Eq.10-12：
#    CAM: AvgPool → Conv-ReLU → Conv-Sigmoid  → [B, C, 1, 1]  (Eq.10)
#    SAM: Conv(k×k,dw)-BN-ReLU → Conv(1×1)-BN-Sigmoid → [B, C, H, W]  (Eq.11)
#    out: x * CAM + x * SAM   (Eq.12)
#
# 注：SAM 使用 k=7 深度可分离卷积对齐论文 Eq.11，
#     P3@80×80 时参数量 = 7*7*C（depthwise），可接受。
# =============================================================================

class PAM(nn.Module):
    """
    论文 Parallel Attention Module。
    CAM 路径（通道注意力）：AvgPool → Conv-ReLU → Conv-Sigmoid
    SAM 路径（空间注意力）：DWConv-BN-ReLU → Conv-BN-Sigmoid
    输出：x * CAM + x * SAM（并联求和，非串行 CBAM）
    """
    def __init__(self, c1: int, c2: int | None = None, k: int = 7):
        super().__init__()
        c1 = _ch(c1)
        c2 = _ch(c2) if c2 is not None else c1
        mid = max(c1 // 16, 8)

        # CAM：全局池化 → bottleneck conv → sigmoid → [B, C, 1, 1] 广播
        self.cam = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c1, 1, bias=False),
            nn.Sigmoid(),
        )

        # SAM：深度可分离空间注意力 → [B, C, H, W]
        self.sam = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=k, padding=k // 2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.cam(x) + x * self.sam(x)   # Eq.12
        return self.out_proj(out)


# =============================================================================
# 3b. BiFPNLite（YOLOv8s-P2-BiFPN-lite）
#     单输入残差精炼，兼容 parse_model patch；alpha=0 初值近似 Identity
# =============================================================================

class BiFPNLite(nn.Module):
    """
    BiFPN-lite residual refinement block.

    YAML: ``- [-1, 1, BiFPNLite, [256]]`` → ``BiFPNLite(c1, c2)`` via parse_model patch.
    放在 FPN/PAN 融合后的 C2f 之后，不做多输入 WeightedAdd。
    """

    def __init__(self, c1: int, c2: int | None = None, k: int = 3):
        super().__init__()
        c1 = _ch(c1)
        c2 = _ch(c2) if c2 is not None else c1
        p = k // 2

        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.dw = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=k, stride=1, padding=p, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.pw = Conv(c2, c2, 1, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        refine = self.pw(self.dw(y))
        return y + torch.tanh(self.alpha) * refine


class HFRes(nn.Module):
    """High-frequency residual refinement for P2/P3 features."""

    def __init__(self, c1: int, c2: int | None = None, k: int = 3):
        super().__init__()
        c1 = _ch(c1)
        c2 = _ch(c2) if c2 is not None else c1
        p = k // 2
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.dw = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=k, stride=1, padding=p, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.pw = Conv(c2, c2, 1, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        hf = self.pw(self.dw(y))
        return y + torch.tanh(self.alpha) * hf


class SGF(nn.Module):
    """Semantic-Guided Fusion for [P2, upsample(P3)] concat features."""

    def __init__(self, c1: int, c2: int | None = None):
        super().__init__()
        c1 = _ch(c1)
        c2 = _ch(c2) if c2 is not None else max(c1 // 2, 8)
        self.c2 = c2
        self.semantic_proj = Conv(max(c1 - c2, 1), c2, 1, 1)
        self.fuse = Conv(c2 * 2, c2, 3, 1)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] <= self.c2:
            p2 = x
            sem = x
        else:
            p2 = x[:, : self.c2, :, :]
            sem = x[:, self.c2 :, :, :]
        sem = self.semantic_proj(sem)
        fused = self.fuse(torch.cat([p2, sem], dim=1))
        return p2 + torch.tanh(self.beta) * fused


class ResPAM(nn.Module):
    """
    残差式并行注意力：out = x + alpha * (attn - x)，attn 与 PAM 相同（CAM+SAM 并联）。
    alpha 初值为 0 → 前向初态等价恒等映射，利于在 YOLOv8s 预训练特征上渐进学习。
    """
    def __init__(self, c1: int, c2: int | None = None, k: int = 7):
        super().__init__()
        c1 = _ch(c1)
        c2 = _ch(c2) if c2 is not None else c1
        mid = max(c1 // 16, 64)#16/64/256/4

        self.cam = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.sam = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=k, padding=k // 2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = x * self.cam(x) + x * self.sam(x)
        out = x + self.alpha * (attn - x)
        return self.out_proj(out)


# =============================================================================
# 3c. AuxDetect — GibbonGuard v2：P2 training-only 辅助监督检测头
# -----------------------------------------------------------------------------
# 主检测头 P3/P4/P5 使用标准 Detect（可继承官方 yolov8s 预训练 Detect 头）；
# 额外挂一个 P2(stride=4) 辅助监督头，仅训练参与联合 loss，推理/验证/导出时
# 完全丢弃（零额外推理开销，部署等价于 3 头）。
# YAML: [[19, 22, 25, 28], 1, AuxDetect, [nc]] → ch=(P2, P3, P4, P5)
# 联合 loss 由 AuxDetectionLoss 计算：L = L_main(P3/P4/P5) + w · L_aux(P2)。
# =============================================================================

if Detect is not None:

    class P2SemanticClsDetect(Detect):
        """P2 box/cls decoupled semantic residual head.

        Keeps standard Detect branches intact. Only P2 cls logits receive an
        optional P3 semantic residual:

            p2_cls = cv3[0](p2) + tanh(alpha) * sem_cls(up(P3))

        alpha is zero-initialized, so the model starts exactly equivalent to
        the original four-head P2 Detect path.
        """

        def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
            super().__init__(nc, reg_max, end2end, ch)
            if len(ch) < 2:
                raise ValueError("P2SemanticClsDetect requires at least P2 and P3 feature channels")
            c_p2, c_p3 = int(ch[0]), int(ch[1])
            self.p2_sem_proj = Conv(c_p3, c_p2, 1, 1)
            self.p2_sem_cls = nn.Sequential(
                DWConv(c_p2, c_p2, 3),
                Conv(c_p2, c_p2, 1),
                nn.Conv2d(c_p2, self.nc, 1),
            )
            self.p2_sem_alpha = nn.Parameter(torch.zeros(1))

        def _p2_semantic_cls(self, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
            p3_up = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
            return self.p2_sem_cls(self.p2_sem_proj(p3_up))

        def forward_head(
            self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
        ) -> dict[str, torch.Tensor]:
            """Standard Detect head, except P2 cls receives a zero-init P3 semantic residual."""
            if box_head is None or cls_head is None:
                return dict()
            bs = x[0].shape[0]
            boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)

            scores = []
            alpha = torch.tanh(self.p2_sem_alpha)
            for i in range(self.nl):
                cls_i = cls_head[i](x[i])
                if i == 0 and len(x) > 1:
                    cls_i = cls_i + alpha * self._p2_semantic_cls(x[0], x[1])
                scores.append(cls_i.view(bs, self.nc, -1))
            scores = torch.cat(scores, dim=-1)
            return dict(boxes=boxes, scores=scores, feats=x)

    class AuxDetect(Detect):
        """主头 P3/P4/P5 + training-only P2 辅助监督头。

        - 训练：forward 返回 dict(boxes, scores, feats, aux)，aux 为 P2 头预测；
          联合 loss 由 AuxDetectionLoss 计算（主 loss + aux_weight × P2 loss）。
        - 推理/验证/导出：仅主 3 头解码，P2 辅助头不参与（零额外推理开销）。
        """

        def __init__(self, nc: int = 80, reg_max: int = 16,
                     end2end: bool = False, ch: tuple = ()):
            assert len(ch) >= 2, "AuxDetect 需要 P2 + 至少一个主尺度"
            main_ch = tuple(ch[1:])      # P3/P4/P5（主检测，继承预训练）
            aux_ch = int(ch[0])          # P2（辅助监督）
            super().__init__(nc, reg_max, end2end, main_ch)
            # 辅助 P2 头（单尺度），结构对齐 Detect.cv2/cv3；永不走 _inference
            c2 = max(16, aux_ch // 4, reg_max * 4)
            c3 = max(aux_ch, min(nc, 100))
            self.aux_cv2 = nn.ModuleList([
                nn.Sequential(Conv(aux_ch, c2, 3), Conv(c2, c2, 3),
                              nn.Conv2d(c2, 4 * reg_max, 1))
            ])
            if self.legacy:
                self.aux_cv3 = nn.ModuleList([
                    nn.Sequential(Conv(aux_ch, c3, 3), Conv(c3, c3, 3),
                                  nn.Conv2d(c3, nc, 1))
                ])
            else:
                self.aux_cv3 = nn.ModuleList([
                    nn.Sequential(
                        nn.Sequential(DWConv(aux_ch, aux_ch, 3), Conv(aux_ch, c3, 1)),
                        nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                        nn.Conv2d(c3, nc, 1),
                    )
                ])

        def _aux_forward(self, p2: torch.Tensor) -> dict:
            bs = p2.shape[0]
            boxes = self.aux_cv2[0](p2).view(bs, 4 * self.reg_max, -1)
            scores = self.aux_cv3[0](p2).view(bs, self.nc, -1)
            return {"boxes": boxes, "scores": scores, "feats": [p2]}

        def forward(self, x):
            main_x = list(x[1:])
            if self.training:
                preds = super().forward(main_x)          # dict(boxes,scores,feats)
                preds["aux"] = self._aux_forward(x[0])    # P2 辅助预测（训练专用）
                return preds
            return super().forward(main_x)                # 推理：仅主 3 头

        def bias_init(self):
            super().bias_init()                           # 主头 bias（P3/P4/P5）
            s = float(self.stride[0]) / 2.0               # P2 stride = P3 stride / 2
            self.aux_cv2[0][-1].bias.data[:] = 1.0
            self.aux_cv3[0][-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


    class AuxDetectionLoss:
        """联合损失：L_main(P3/P4/P5) + aux_weight · L_aux(P2)。

        推理/验证时 preds 不含 'aux' 键 → 自动退化为标准主 loss。
        日志仍只显示主头的 (box, cls, dfl)，aux 仅叠加到反传的总 loss。
        """

        def __init__(self, model, aux_weight: float = 0.5):
            from ultralytics.utils.loss import v8DetectionLoss
            from ultralytics.utils.tal import TaskAlignedAssigner
            self.aux_weight = float(aux_weight)
            self.main = v8DetectionLoss(model)
            self.hyp = self.main.hyp
            # 辅助 loss：复用 v8DetectionLoss，但 stride/assigner 改为 P2(stride=P3/2)
            m = model.model[-1]
            aux_stride = (m.stride[:1] / 2.0).clone()
            self.aux = v8DetectionLoss(model)
            self.aux.stride = aux_stride
            self.aux.assigner = TaskAlignedAssigner(
                topk=10, num_classes=self.aux.nc, alpha=0.5, beta=6.0,
                stride=aux_stride.tolist(), topk2=None,
            )

        def __call__(self, preds, batch):
            if isinstance(preds, (list, tuple)):
                preds = preds[1]
            main_total, main_items = self.main(preds, batch)
            aux = preds.get("aux") if isinstance(preds, dict) else None
            if aux is not None:
                aux_total, _ = self.aux(aux, batch)
                return main_total + self.aux_weight * aux_total, main_items
            return main_total, main_items


    class GibbonQualityDetectionLoss:
        """YOLOv8 detection loss + small-object NWD auxiliary localization term.

        This is intentionally conservative:
        - reuses the original v8DetectionLoss and TaskAlignedAssigner
        - does not modify Detect or bbox/DFL internals
        - adds a low-weight NWD term only for small positive samples
        """

        def __init__(
            self,
            model,
            lambda_nwd: float | None = None,
            nwd_c: float | None = None,
            small_tau: float | None = None,
            ramp_start: int | None = None,
            ramp_end: int | None = None,
        ):
            from ultralytics.utils.loss import v8DetectionLoss

            self.main = v8DetectionLoss(model)
            self.hyp = self.main.hyp
            self.model = model
            self.lambda_nwd = float(os.environ.get("RF_GQL_LAMBDA_NWD", lambda_nwd if lambda_nwd is not None else 0.10))
            self.nwd_c = float(os.environ.get("RF_GQL_NWD_C", nwd_c if nwd_c is not None else 20.0))
            self.small_tau = float(os.environ.get("RF_GQL_SMALL_TAU", small_tau if small_tau is not None else 0.05))
            self.ramp_start = int(os.environ.get("RF_GQL_RAMP_START", ramp_start if ramp_start is not None else 5))
            self.ramp_end = int(os.environ.get("RF_GQL_RAMP_END", ramp_end if ramp_end is not None else 20))

        @staticmethod
        def _xyxy_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
            x1, y1, x2, y2 = box.unbind(-1)
            w = (x2 - x1).clamp(min=1e-6)
            h = (y2 - y1).clamp(min=1e-6)
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            return torch.stack((cx, cy, w, h), dim=-1)

        def _ramp_weight(self) -> float:
            trainer = getattr(self.model, "trainer", None)
            epoch = getattr(trainer, "epoch", None)
            if epoch is None:
                epoch = getattr(self.model, "epoch", None)
            if epoch is None:
                return self.lambda_nwd
            epoch = int(epoch)
            if epoch < self.ramp_start:
                return 0.0
            if epoch >= self.ramp_end:
                return self.lambda_nwd
            span = max(1, self.ramp_end - self.ramp_start)
            return self.lambda_nwd * float(epoch - self.ramp_start) / float(span)

        def _small_nwd_loss(self, preds, assigned) -> torch.Tensor:
            fg_mask, _target_gt_idx, target_bboxes, anchor_points, stride_tensor = assigned
            if not bool(fg_mask.sum()):
                return target_bboxes.sum() * 0.0

            pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
            pred_bboxes = self.main.bbox_decode(anchor_points, pred_distri) * stride_tensor
            pred_fg = pred_bboxes[fg_mask]
            target_fg = target_bboxes[fg_mask]
            if pred_fg.numel() == 0:
                return target_bboxes.sum() * 0.0

            p = self._xyxy_to_cxcywh(pred_fg)
            g = self._xyxy_to_cxcywh(target_fg)
            diff = p - g
            w2 = diff[:, 0].pow(2) + diff[:, 1].pow(2) + (diff[:, 2].pow(2) + diff[:, 3].pow(2)) * 0.25
            nwd_loss = 1.0 - torch.exp(-torch.sqrt(w2 + 1e-9) / self.nwd_c)

            wh = (target_fg[:, 2:4] - target_fg[:, 0:2]).clamp(min=1e-6)
            # target_bboxes are in input-image pixel coordinates.
            img_hw = torch.as_tensor(preds["feats"][0].shape[2:], device=target_fg.device, dtype=target_fg.dtype) * self.main.stride[0]
            img_area = (img_hw[0] * img_hw[1]).clamp(min=1.0)
            area_ratio = (wh[:, 0] * wh[:, 1]) / img_area
            w_small = ((self.small_tau - area_ratio) / self.small_tau).clamp(min=0.0, max=1.0)
            denom = w_small.sum().clamp(min=1.0)
            return (nwd_loss * w_small).sum() / denom

        def __call__(self, preds, batch):
            preds = self.main.parse_output(preds)
            batch_size = preds["boxes"].shape[0]
            assigned, loss, _ = self.main.get_assigned_targets_and_loss(preds, batch)
            nwd_w = self._ramp_weight()
            if nwd_w > 0:
                loss[0] = loss[0] + nwd_w * self._small_nwd_loss(preds, assigned)
            return loss * batch_size, loss.detach()


# =============================================================================
# 4. RST（Residual Swin Transformer）
#    论文 Section 3.4 / 图2 / Eq.1-5
#
#    FIX-5:  SW-MSA mask 正确传入 WindowMSA.forward
#    FIX-6:  WindowMSA eager init，EMA 参数完整
#    FIX-7:  num_heads 自动调整至整除 dim
#    FIX-10: rel_pos 异常只捕获设备/形状不匹配，不静默吞掉逻辑错误
#    FIX-11: MLP 加 dropout（p=0.0 默认，行为不变）
#    FIX-12: 注意力 + proj 加 dropout（p=0.0 默认）
#    FIX-15: rel_pos bias 与 attn 强制对齐 device
# =============================================================================

class RelativePositionBias(nn.Module):
    """
    论文 Eq.9 可学习相对位置偏置。
    参数表大小：(2Wh-1)*(2Ww-1) × num_heads，每次 forward 做 index gather。
    参数随梯度更新，不可缓存 forward 结果。
    """
    def __init__(self, window_size: tuple[int, int], num_heads: int):
        super().__init__()
        self.window_size = window_size
        Wh, Ww = window_size
        self.table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads)
        )
        nn.init.trunc_normal_(self.table, std=0.02)

        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        grid     = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # [2,Wh,Ww]
        flat     = grid.flatten(1)                                                   # [2, Wh*Ww]
        rel      = flat[:, :, None] - flat[:, None, :]                              # [2, N, N]
        rel      = rel.permute(1, 2, 0).contiguous()                                # [N, N, 2]
        rel[:, :, 0] += Wh - 1
        rel[:, :, 1] += Ww - 1
        rel[:, :, 0] *= 2 * Ww - 1
        self.register_buffer("index", rel.sum(-1).long())                           # [N, N]

    def forward(self) -> torch.Tensor:
        N   = self.window_size[0] * self.window_size[1]
        idx = self.index.view(-1)
        # FIX-15：确保 table 与 index 在同一 device（model.to(device) 时 buffer 自动迁移）
        tbl = self.table.to(self.index.device)
        b   = tbl[idx].view(N, N, -1)          # [N, N, nH]
        return b.permute(2, 0, 1).contiguous()  # [nH, N, N]


class WindowMSA(nn.Module):
    """
    Window Multi-head Self-Attention（W-MSA 或 SW-MSA）。
    对应论文 Eq.6-9。

    FIX-5:  forward 接受并应用 attn_mask（SW-MSA 跨区域屏蔽）
    FIX-6:  本模块在父 SwinTransformerBlock2d.__init__ 中实例化（eager），EMA 参数完整
    FIX-7:  自动调整 num_heads 使 dim 整除
    FIX-10: rel_pos 只捕获 RuntimeError（形状/设备），不静默吞 ValueError 等逻辑错误
    FIX-12: attention dropout + proj dropout
    """
    def __init__(self, dim: int, window_size: tuple[int, int],
                 num_heads: int = 4, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        # FIX-7：调整 num_heads 使 dim 严格整除
        h = min(num_heads, dim)
        while h > 1 and dim % h != 0:
            h -= 1
        self.num_heads = h
        self.head_dim  = dim // h
        self.scale     = self.head_dim ** -0.5
        self.dim       = dim
        self.window_size = window_size

        self.qkv      = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)   # FIX-12
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)   # FIX-12
        self.rel_pos   = RelativePositionBias(window_size, self.num_heads)

        # SW-MSA mask 缓存：key = (H_padded, W_padded, device_str)
        # 注：不是 Buffer，不随 .to(device) 迁移；key 含 device 字符串，
        #     迁移后首次调用会重新生成并缓存在新 device 上
        self._mask_cache: dict[tuple, torch.Tensor] = {}

    def _get_shift_mask(self, H: int, W: int,
                        device: torch.device) -> torch.Tensor | None:
        """生成 SW-MSA 跨区域屏蔽 mask [nW, N, N]；W-MSA 返回 None。"""
        key = (H, W, str(device))
        if key in self._mask_cache:
            return self._mask_cache[key]

        Wh, Ww = self.window_size
        sh, sw = Wh // 2, Ww // 2
        img   = torch.zeros(1, H, W, 1, device=device)
        cnt   = 0
        for hs in (slice(0, -Wh), slice(-Wh, -sh), slice(-sh, None)):
            for ws in (slice(0, -Ww), slice(-Ww, -sw), slice(-sw, None)):
                img[:, hs, ws, :] = cnt
                cnt += 1
        wins = _window_partition_hw(img, Wh, Ww).view(-1, Wh * Ww)  # [nW, N]
        mask = wins.unsqueeze(1) - wins.unsqueeze(2)                  # [nW, N, N]
        mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
        self._mask_cache[key] = mask
        return mask

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x        : [nW*B, N, C]，N = Wh*Ww
        attn_mask: [nW, N, N]，SW-MSA 时由 SwinTransformerBlock2d 传入
        FIX-5    : 将 mask 加到 logit 上，cyclic shift 才真正有效
        """
        dtype_in = x.dtype
        x  = x.float()
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B_, nH, N, hd]

        attn = (q @ k.transpose(-2, -1)) * self.scale    # [B_, nH, N, N]

        # 相对位置偏置 Eq.9
        # FIX-10: 只捕获 RuntimeError（device/shape 不符），保留逻辑错误的抛出
        try:
            rpb  = self.rel_pos()                         # [nH, N, N]
            attn = attn + rpb.to(attn.device).unsqueeze(0)
        except RuntimeError as e:
            warnings.warn(f"[RST] RelativePositionBias 跳过（{e}）", stacklevel=2)

        # FIX-5：应用 SW-MSA 跨区域屏蔽 mask
        if attn_mask is not None:
            nW  = attn_mask.shape[0]
            B   = B_ // nW
            attn = attn.view(B, nW, self.num_heads, N, N)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0).to(attn.device, attn.dtype)
            attn = attn.view(B_, self.num_heads, N, N)

        attn = attn.clamp(-50.0, 50.0).softmax(dim=-1)
        attn = self.attn_drop(attn)                       # FIX-12
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x    = self.proj_drop(self.proj(x))               # FIX-12
        return x.to(dtype_in)


def _window_partition_hw(x: torch.Tensor, Wh: int, Ww: int) -> torch.Tensor:
    """[B, H, W, C] → [B*nW, Wh*Ww, C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // Wh, Wh, W // Ww, Ww, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, Wh * Ww, C)


def _window_reverse_hw(wins: torch.Tensor, Wh: int, Ww: int,
                       H: int, W: int) -> torch.Tensor:
    """[B*nW, Wh*Ww, C] → [B, H, W, C]"""
    B = wins.shape[0] // (H // Wh * W // Ww)
    x = wins.view(B, H // Wh, W // Ww, Wh, Ww, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class SwinTransformerBlock2d(nn.Module):
    """
    完整 Swin Transformer Block：
      LN → (W/SW)-MSA → +residual₁ → LN → MLP → +residual₂
    对应论文 Eq.1/2（W-MSA）或 Eq.3/4（SW-MSA）。

    FIX-6  : WindowMSA 在 __init__ 中直接实例化（eager），EMA 参数不缺失
    FIX-11 : MLP 加 Dropout（p=0.0 默认）
    """
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 7,
                 mlp_ratio: float = 4.0, shift: bool = False,
                 qkv_bias: bool = True,
                 attn_drop: float = 0.0,   # FIX-12
                 proj_drop: float = 0.0,   # FIX-12
                 mlp_drop:  float = 0.0):  # FIX-11
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.shift       = shift

        self.norm1 = nn.LayerNorm(dim)   # 论文标准：LN（非 BN）
        self.norm2 = nn.LayerNorm(dim)

        # FIX-6：eager init，EMA 创建前参数已存在
        self.attn = WindowMSA(
            dim, (window_size, window_size),
            num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(mlp_drop),          # FIX-11
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(mlp_drop),          # FIX-11
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W]；内部转 float32 保证 AMP 安全。"""
        dtype_in = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        win = self.window_size

        # padding 保证 H/W 可整除 window_size
        pad_h = (win - H % win) % win
        pad_w = (win - W % win) % win

        x_hwc = x.permute(0, 2, 3, 1).contiguous()   # [B, H, W, C]
        if pad_h or pad_w:
            x_hwc = F.pad(x_hwc, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x_hwc.shape[1], x_hwc.shape[2]

        # cyclic shift（SW-MSA）
        sh = sw = (win // 2) if self.shift else 0
        x_shifted = torch.roll(x_hwc, shifts=(-sh, -sw), dims=(1, 2)) if self.shift else x_hwc

        # FIX-5：生成并传入 shift mask
        attn_mask = self.attn._get_shift_mask(Hp, Wp, x.device) if self.shift else None

        # LN → 窗口划分 → MSA → 窗口还原
        x_ln  = self.norm1(x_shifted)
        x_win = _window_partition_hw(x_ln, win, win)
        x_win = self.attn(x_win, attn_mask=attn_mask)
        x_win = _window_reverse_hw(x_win, win, win, Hp, Wp)

        # 反 cyclic shift
        if self.shift:
            x_win = torch.roll(x_win, shifts=(sh, sw), dims=(1, 2))

        x_hwc = x_hwc + x_win                          # residual₁ Eq.1/Eq.3
        x_hwc = x_hwc + self.mlp(self.norm2(x_hwc))   # residual₂ Eq.2/Eq.4

        if pad_h or pad_w:
            x_hwc = x_hwc[:, :H, :W, :].contiguous()

        return x_hwc.permute(0, 3, 1, 2).contiguous().to(dtype_in)

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, window_size={self.window_size}, "
                f"shift={self.shift}, heads={self.attn.num_heads}")


# ── 向后兼容别名（旧版权重 pickle 使用的类名）──────────────────────────────
# FIX-22: 旧版 ultralytics_rainforest.py 中 SwinTransformerBlock2d 曾命名为
#         SwinBlock2d，torch.load 反序列化时按旧类名查找。添加别名避免 AttributeError。
SwinBlock2d = SwinTransformerBlock2d


class RST(nn.Module):
    """
    Residual Swin Transformer（论文图2 / Eq.1-5）：
      F_W  = W-MSA block(F_in)      — 含两个小残差（Eq.1,2）
      F_SW = SW-MSA block(F_W)      — 含两个小残差（Eq.3,4）
      F_out = proj(F_SW + F_in)     — 大残差（Eq.5）

    注意：所有 BUG 通过修复 SwinTransformerBlock2d / WindowMSA 解决。
    RST 本身只负责：大残差 + 可选的 proj（c1≠c2 时）。
    """
    def __init__(self, c1: int, c2: int | None = None,
                 window_size: int = 7, num_heads: int = 4,
                 mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 mlp_drop: float = 0.0):
        super().__init__()
        c1 = _ch(c1)
        c2 = _ch(c2) if c2 is not None else c1
        # 上层：W-MSA（不 shift）
        self.block_w  = SwinTransformerBlock2d(
            c1, num_heads, window_size, mlp_ratio, shift=False,
            attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop)
        # 下层：SW-MSA（shift）
        self.block_sw = SwinTransformerBlock2d(
            c1, num_heads, window_size, mlp_ratio, shift=True,
            attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop)
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x                           # F_RST_in，用于 Eq.5 大残差
        x = self.block_w(x.float())            # Eq.1 Eq.2（W-MSA）
        x = self.block_sw(x)                   # Eq.3 Eq.4（SW-MSA）
        x = x.to(identity.dtype) + identity   # Eq.5：大残差
        return self.proj(x)


class ST(nn.Module):
    """
    Swin Transformer（无大残差），用于论文表4消融实验 '+ST' 行。
    与 RST 的区别：缺少 Eq.5 的大残差，仅有两个 Swin Block。
    """
    def __init__(self, c1: int, c2: int | None = None,
                 window_size: int = 7, num_heads: int = 4,
                 mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 mlp_drop: float = 0.0):
        super().__init__()
        c1 = _ch(c1)
        c2 = _ch(c2) if c2 is not None else c1
        self.block_w  = SwinTransformerBlock2d(
            c1, num_heads, window_size, mlp_ratio, shift=False,
            attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop)
        self.block_sw = SwinTransformerBlock2d(
            c1, num_heads, window_size, mlp_ratio, shift=True,
            attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop)
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype_in = x.dtype                  # BUG-21 FIX: 记录输入 dtype
        x = self.block_w(x.float())
        x = self.block_sw(x)
        return self.proj(x).to(dtype_in)    # 还原，与 RST 对称


# =============================================================================
# 5. 注册到 Ultralytics
# =============================================================================
# FIX-19 根因：
#   新版 Ultralytics parse_model 通过 getattr(ultralytics.nn.modules, m_name)
#   查找模块。只注册到 tasks 时 m 仍是字符串，elif 条件 m.__name__ 不匹配，
#   实际走 else 分支，args[0] 被当作 c1 → C3k2_AdaConv(128, False, ...) → ERROR。
# 修复（三层保险）：
#   1. 同时注册到 nn.modules / nn.modules.block（覆盖所有查找路径）
#   2. elif 条件兼容 class/string 两种形式（_get_module_name）
#   3. 策略 B：将自定义模块加入 parse_model 源码 C3k2 所在的外层 set

_CUSTOM_NAMES = frozenset(
    ["C3k2_AdaConv", "C2f_AdaConv", "MSCB", "PAM", "ResPAM", "HFRes", "SGF", "BiFPNLite", "RST", "ST"]
)


def _get_module_name(m: object) -> str:
    """class 和 string 两种形式都能返回模块名。"""
    if isinstance(m, str):
        return m
    return getattr(m, "__name__", getattr(m, "__qualname__", ""))


def _build_parse_model_patch_src(indent: str) -> str:
    """
    构建注入到 parse_model 的 elif 分支源码。

    BUG-20 FIX: 删除原有的"pre-scaled 检测"（c2 <= max_channels*width 不缩放）。
    该检测在边界值（如 YAML=512, max_channels*width=512）时误判为"已缩放"，
    导致 MSCB/PAM/RST 输出通道翻倍（512 而非 256）。
    策略A 始终执行 make_divisible 缩放，与 C3k2 标准处理保持一致。
    策略B 作为独立保险仍保留。
    """
    i4 = indent + "    "
    i8 = indent + "        "
    NL = "\n"
    return (
        indent + "elif _get_module_name(m) in _CUSTOM_NAMES and args:" + NL
        + i4 + "c2 = args[0]" + NL
        + i4 + "if isinstance(c2, (int, float)) and c2 != nc:" + NL
        + i8 + "# BUG-20 FIX: 始终缩放，不做 pre-scaled 检测（边界值会误判）" + NL
        + i8 + "c2 = make_divisible(min(c2, max_channels) * width, 8)" + NL
        + i4 + "args = [ch[f], c2, *args[1:]]" + NL
    )


def _rainforest_parse_model_patch() -> None:
    """
    双策略 patch parse_model，覆盖全部 Ultralytics 版本：
      策略 A：在 else: c2 = ch[f] 前注入 elif（兼容 class/string 形式的 m）
      策略 B：将自定义模块加入 C3k2 外层 set（首次出现，不触碰 n-insertion 子集）
    """
    if tasks is None:
        return

    import inspect
    try:
        from ultralytics.utils.ops import make_divisible
    except ImportError:
        try:
            from ultralytics.utils.torch_utils import make_divisible  # type: ignore[no-redef]
        except ImportError:
            warnings.warn("[Rainforest] parse_model patch 跳过：未找到 make_divisible", stacklevel=2)
            return

    orig = tasks.parse_model
    if getattr(orig, "_rf_patched", False):
        return

    try:
        src = inspect.getsource(orig)
    except Exception as e:
        warnings.warn(f"[Rainforest] 无法读取 parse_model 源码（{e}），patch 跳过", stacklevel=2)
        return

    new_src = src

    # ── 策略 A ────────────────────────────────────────────────────────────────
    ELSE_CANDIDATES = [
        "        else:\n            c2 = ch[f]",
        "        else:\n                c2 = ch[f]",
    ]
    strategy_a_ok = False
    for else_line in ELSE_CANDIDATES:
        if else_line in new_src:
            indent    = else_line.split("else:")[0]
            injection = _build_parse_model_patch_src(indent) + else_line
            new_src   = new_src.replace(else_line, injection, 1)
            strategy_a_ok = True
            break
    if not strategy_a_ok:
        warnings.warn("[Rainforest] 策略 A 未找到注入点，仅策略 B 生效", stacklevel=2)

    # ── 策略 B：把自定义模块加入 C3k2 外层 set 的第一次出现 ───────────────────
    _add = ", ".join(sorted(_CUSTOM_NAMES))
    if "C3k2," in new_src:
        new_src = new_src.replace("C3k2,", f"C3k2, {_add},", 1)

    # ── Detect-like heads：加入 Detect head 集合，使其走 args.extend([reg_max,end2end,ch]) 分支 ──
    _aux = globals().get("AuxDetect")
    _sem = globals().get("P2SemanticClsDetect")
    if _aux is not None:
        _anchor = "                Detect,\n"
        if _anchor in new_src:
            new_src = new_src.replace(_anchor, _anchor + "                AuxDetect,\n", 1)
        new_src = new_src.replace("if m in {Detect,", "if m in {Detect, AuxDetect,", 1)
    if _sem is not None:
        _anchor = "                Detect,\n"
        if _anchor in new_src:
            new_src = new_src.replace(_anchor, _anchor + "                P2SemanticClsDetect,\n", 1)
        new_src = new_src.replace("if m in {Detect,", "if m in {Detect, P2SemanticClsDetect,", 1)
        new_src = new_src.replace("if m in {Detect, AuxDetect,", "if m in {Detect, AuxDetect, P2SemanticClsDetect,", 1)

    # ── exec ─────────────────────────────────────────────────────────────────
    try:
        globs = dict(orig.__globals__)
        globs["_CUSTOM_NAMES"]    = _CUSTOM_NAMES
        globs["_get_module_name"] = _get_module_name
        if _aux is not None:
            globs["AuxDetect"] = _aux
        if _sem is not None:
            globs["P2SemanticClsDetect"] = _sem
        exec(compile(new_src, "<rf_parse_patch>", "exec"), globs)
        fn = globs.get("parse_model")
        if fn is None:
            raise ValueError("exec 后未找到 parse_model")
        fn._rf_patched = True
        tasks.parse_model = fn
        mode = "A+B" if strategy_a_ok else "B only"
        print(f"[Rainforest] ✅ parse_model patch 成功（策略 {mode}）")
    except Exception as e:
        warnings.warn(f"[Rainforest] parse_model patch exec 失败（{e}）", stacklevel=2)


def register_rainforest_modules() -> None:
    """
    FIX-19：同时注册到三个位置，覆盖不同版本的 parse_model 查找路径：
      ultralytics.nn.tasks         → 旧版 globals()['name']
      ultralytics.nn.modules       → 新版 getattr(modules, 'name')
      ultralytics.nn.modules.block → 部分版本的 block 子模块
    """
    if tasks is None:
        warnings.warn("[Rainforest] Ultralytics 未安装，注册跳过", stacklevel=2)
        return

    _MODULES = {
        "C3k2_AdaConv": C3k2_AdaConv,
        "C2f_AdaConv":  C2f_AdaConv,
        "MSCB":         MSCB,
        "PAM":          PAM,
        "ResPAM":       ResPAM,
        "HFRes":        HFRes,
        "SGF":          SGF,
        "BiFPNLite":    BiFPNLite,
        "RST":          RST,
        "ST":           ST,
    }

    # 注册到 tasks（parse_model globals，旧版）
    for name, cls in _MODULES.items():
        setattr(tasks, name, cls)

    # 注册到 nn.modules 和子模块（新版查找路径）
    try:
        from ultralytics.nn import modules as _nn_modules
        for name, cls in _MODULES.items():
            setattr(_nn_modules, name, cls)
        try:
            from ultralytics.nn.modules import block as _nn_block
            for name, cls in _MODULES.items():
                setattr(_nn_block, name, cls)
        except ImportError:
            pass
        try:
            from ultralytics.nn.modules import head as _nn_head
            for name, cls in _MODULES.items():
                setattr(_nn_head, name, cls)
        except ImportError:
            pass
    except ImportError:
        pass

    # AuxDetect 注册到各查找路径（在 parse_model patch 前，使 head set 替换可用）
    _aux = globals().get("AuxDetect")
    if _aux is not None:
        setattr(tasks, "AuxDetect", _aux)
        try:
            from ultralytics.nn import modules as _nn_modules2
            setattr(_nn_modules2, "AuxDetect", _aux)
            from ultralytics.nn.modules import head as _nn_head2
            setattr(_nn_head2, "AuxDetect", _aux)
        except ImportError:
            pass

    # P2SemanticClsDetect 注册为 Detect-like head（非普通 neck 模块）
    _sem = globals().get("P2SemanticClsDetect")
    if _sem is not None:
        setattr(tasks, "P2SemanticClsDetect", _sem)
        try:
            from ultralytics.nn import modules as _nn_modules3
            setattr(_nn_modules3, "P2SemanticClsDetect", _sem)
            from ultralytics.nn.modules import head as _nn_head3
            setattr(_nn_head3, "P2SemanticClsDetect", _sem)
        except ImportError:
            pass

    _rainforest_parse_model_patch()

    # init_criterion：AuxDetect 用联合辅助监督；普通 Detect 可通过环境变量启用 GibbonQualityLoss。
    if _aux is not None:
        _orig_ic = tasks.DetectionModel.init_criterion
        if not getattr(_orig_ic, "_rf_aux", False):
            def _aux_init_criterion(self, _orig=_orig_ic, _AuxDetect=_aux):
                if isinstance(self.model[-1], _AuxDetect):
                    return AuxDetectionLoss(self)
                if os.environ.get("RF_GIBBON_QUALITY_LOSS", "0") == "1":
                    return GibbonQualityDetectionLoss(self)
                return _orig(self)
            _aux_init_criterion._rf_aux = True
            tasks.DetectionModel.init_criterion = _aux_init_criterion
            print("[Rainforest-Aux] ✅ init_criterion 已支持 AuxDetectionLoss / GibbonQualityDetectionLoss")

    print(f"[Rainforest] ✅ 已注册：{', '.join(_MODULES)}"
          + (" + AuxDetect" if _aux is not None else "")
          + (" + P2SemanticClsDetect" if _sem is not None else "")
          + " + GibbonQualityDetectionLoss")


# 导入即注册
register_rainforest_modules()


class RainforestLowLightTransform:
    """Pickle-safe low-light image transform for Windows DataLoader workers."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, image):
        import random as _random
        import numpy as _np

        c = self.cfg
        f = image.astype(_np.float32) / 255.0
        f = _np.power(_np.clip(f, 0, 1), _random.uniform(*c["gamma"]))
        a = _random.uniform(*c["contrast"])
        b = _random.uniform(*c["bright"])
        m = float(f.mean())
        f = (f - m) * a + m + b
        f = _np.clip(f, 0, 1)
        if _random.random() < c["flare_p"]:
            h, w = f.shape[:2]
            cx = _random.randint(0, w - 1)
            cy = _random.randint(0, max(1, int(h * 0.5)))
            rad = _random.uniform(0.20, 0.45) * max(h, w)
            yy, xx = _np.ogrid[:h, :w]
            halo = _np.clip(1.0 - _np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rad, 0, 1)[..., None] ** 2
            f = _np.clip(f + halo * _random.uniform(0.30, 0.70), 0, 1)
        return (f * 255.0).astype(_np.uint8)


class RainforestLowLightCompose:
    """Callable object matching Albumentations' image=... API; pickle-safe."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.t = RainforestLowLightTransform(cfg)

    def __call__(self, image=None, **kwargs):
        import random as _random

        if image is None:
            return kwargs
        if _random.random() > self.cfg["p"]:
            return {"image": image, **kwargs}
        return {"image": self.t(image), **kwargs}


def _install_lowlight_aug_patch():
    """逆光/低对比度数据增强（env LOWLIGHT_AUG 控制，默认关闭）。

    albumentations 未安装，故用 cv2/numpy 自实现，patch 进 ultralytics 增强管线中
    Albumentations 占位（augment.py 中 v8_transforms 已调用 Albumentations(p=1.0)）。
    纯像素级（gamma/亮度/对比度/眩光），不改 bbox。
    强度: LOWLIGHT_AUG=mild|medium|strong（或 1/true 等价 medium）。
    """
    import os
    level = os.environ.get("LOWLIGHT_AUG", "").strip().lower()
    if level in ("", "0", "false", "off", "none"):
        return
    if level in ("1", "true", "on", "yes"):
        level = "medium"
    presets = {
        "mild":   dict(p=0.30, gamma=(0.70, 1.50), bright=(-0.25, 0.20), contrast=(0.80, 1.10), flare_p=0.05),
        "medium": dict(p=0.50, gamma=(0.55, 1.70), bright=(-0.40, 0.30), contrast=(0.65, 1.15), flare_p=0.12),
        "strong": dict(p=0.65, gamma=(0.45, 1.90), bright=(-0.55, 0.40), contrast=(0.55, 1.25), flare_p=0.20),
    }
    cfg = presets.get(level, presets["medium"])

    try:
        import ultralytics.data.augment as _aug
    except Exception as e:
        print(f"[Rainforest-LowLight] ✗ 无法 patch augment: {e}")
        return

    def _ll_init(self, p=1.0, transforms=None):
        self.p = float(p)
        self.transform = RainforestLowLightCompose(cfg)   # 兼容原始 __call__: self.transform(image=...)["image"]
        self.contains_spatial = False
        self._ll = cfg

    def _ll_call(self, labels):
        import random as _random

        if _random.random() > self._ll["p"]:
            return labels
        im = labels.get("img")
        if im is None or getattr(im, "ndim", 0) != 3 or im.shape[2] != 3:
            return labels
        labels["img"] = RainforestLowLightTransform(self._ll)(im)
        return labels

    _aug.Albumentations.__init__ = _ll_init
    _aug.Albumentations.__call__ = _ll_call
    print(f"[Rainforest-LowLight] ✅ 逆光增强已开启 (LOWLIGHT_AUG={level}): "
          f"p={cfg['p']} gamma={cfg['gamma']} bright={cfg['bright']} contrast={cfg['contrast']} flare_p={cfg['flare_p']}")


_install_lowlight_aug_patch()