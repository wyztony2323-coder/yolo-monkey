import sys
sys.path.insert(0, r"D:\college\college3\monkey\yolo\finalmodel\ultralytics_local")
import argparse
import copy
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
import random
import shutil
import time
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.nn import tasks

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ==================== NVTX 性能分析支持 ====================
# 用于 Nsight Systems 时间线标注（RTX 5060 / Blackwell 架构）
# 使用方式（标准）：
#   nsys profile --trace=cuda,nvtx --duration=300 -o yolo11_train python train_k.py --model_type rainforest --single_fold
#
# 使用方式（轻量级，推荐）：
#   .\run_nsys_lightweight.ps1
#
# 使用方式（NVTX 范围捕获，最小文件）：
#   .\run_nsys_nvtx_range.ps1
#
# 注意：
#   - 如果 GPU 被占用，可以移除 --gpu-metrics-devices 参数
#   - 如果文件过大，使用 --duration=300 限制采样时间（推荐）
def nvtx_range_push(name):
    """推送 NVTX 范围（如果 CUDA 可用）"""
    try:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
    except Exception:
        pass

def nvtx_range_pop():
    """弹出 NVTX 范围（如果 CUDA 可用）"""
    try:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
    except Exception:
        pass


# ==================== [实时"数据流/算子时间线"可视化：PyTorch Profiler -> TensorBoard] ====================
def install_live_profiler_ultralytics(
    yolo_model,
    logdir="runs/liveflow",
    enable_nvtx_batch=True,
    enable_profiler=True,
    prof_wait=30,
    prof_warmup=2,
    prof_active=10,
    prof_repeat=1,
    enable_grad_clip=False,
    grad_clip_max_norm=10.0,
):
    """
    安装实时性能分析器到 Ultralytics YOLO 模型。

    重要：默认不再附带梯度裁剪。
    这样 baseline 模型在不开启额外训练干预时，保持纯粹 baseline 训练路径；
    只有显式要求时才会打开 grad clip。
    """
    import os
    import time
    os.makedirs(logdir, exist_ok=True)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(logdir)
        abs_logdir = os.path.abspath(logdir)
        print(f"[TensorBoard] ✅ SummaryWriter 已创建")
        print(f"[TensorBoard] 日志目录: {abs_logdir}")
        print(f"[TensorBoard] 启动命令: tensorboard --logdir {os.path.dirname(abs_logdir)} --port 6006")
        print(f"[TensorBoard] 或者: tensorboard --logdir {abs_logdir} --port 6006")
        print(f"[TensorBoard] ⚠️  注意：需要等待训练开始后，数据才会写入。如果 TensorBoard 显示 'No dashboards'，请等待几个 batch。")
    except ImportError:
        print("[TensorBoard] ⚠️  警告：tensorboard 未安装，无法记录标量数据")
        print("[TensorBoard] 安装命令: pip install tensorboard")
        writer = None
    except Exception as e:
        print(f"[TensorBoard] ⚠️  警告：创建 SummaryWriter 失败: {e}")
        writer = None

    prof = None
    if enable_profiler:
        try:
            from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
            prof_logdir = os.path.join(logdir, "tb_profile")
            os.makedirs(prof_logdir, exist_ok=True)
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=prof_wait, warmup=prof_warmup, active=prof_active, repeat=prof_repeat),
                on_trace_ready=tensorboard_trace_handler(prof_logdir),
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
            )
            prof.__enter__()
            print(f"[Profiler] 已启用，采样窗口: wait={prof_wait}, warmup={prof_warmup}, active={prof_active}")
        except ImportError as e:
            print(f"[Profiler] ⚠️  警告：torch.profiler 不可用：{e}")
        except Exception as e:
            print(f"[Profiler] ⚠️  警告：初始化失败：{e}")

    state = {"step": 0, "t0": time.time()}

    def _safe_float(x, default=None):
        try:
            return float(x)
        except Exception:
            return default

    def on_train_batch_start(trainer):
        if enable_nvtx_batch and torch.cuda.is_available():
            epoch = getattr(trainer, "epoch", -1)
            batch_i = getattr(trainer, "batch_i", -1)
            torch.cuda.nvtx.range_push(f"batch e{epoch} i{batch_i}")

    def on_train_batch_end(trainer):
        if enable_grad_clip:
            try:
                if hasattr(trainer, 'model') and trainer.model is not None:
                    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=grad_clip_max_norm)
            except Exception:
                pass

        step = state["step"]

        if writer is not None:
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                writer.add_scalar("gpu/mem_alloc_GB", mem_alloc, step)
                writer.add_scalar("gpu/mem_reserved_GB", mem_reserved, step)

            loss_items = getattr(trainer, "loss_items", None)
            if loss_items is not None:
                try:
                    if hasattr(loss_items, "detach"):
                        li = loss_items.detach().float().cpu().tolist()
                    else:
                        li = list(loss_items)
                    for k, v in enumerate(li):
                        fv = _safe_float(v)
                        if fv is not None:
                            writer.add_scalar(f"loss/item_{k}", fv, step)
                except Exception:
                    pass

            lr = getattr(trainer, "lr", None)
            if isinstance(lr, dict):
                for k, v in lr.items():
                    fv = _safe_float(v)
                    if fv is not None:
                        writer.add_scalar(f"lr/{k}", fv, step)
            else:
                fv = _safe_float(lr)
                if fv is not None:
                    writer.add_scalar("lr", fv, step)

            t = time.time()
            dt = t - state["t0"]
            state["t0"] = t
            writer.add_scalar("time/step_s", dt, step)

        if prof is not None:
            try:
                prof.step()
            except Exception:
                pass

        if enable_nvtx_batch and torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

        state["step"] += 1

    add_cb = getattr(yolo_model, "add_callback", None)
    if callable(add_cb):
        add_cb("on_train_batch_start", on_train_batch_start)
        add_cb("on_train_batch_end", on_train_batch_end)
        print("[Profiler] ✅ 回调已注册到 Ultralytics 模型")
    else:
        print("[Profiler] ⚠️  警告：当前 Ultralytics 版本未发现 model.add_callback；无法挂载实时 profiler")

    def close():
        try:
            if writer is not None:
                writer.flush()
                writer.close()
                print("[TensorBoard] ✅ 已关闭 SummaryWriter")
        except Exception:
            pass
        finally:
            if prof is not None:
                try:
                    prof.__exit__(None, None, None)
                    print("[Profiler] ✅ 已关闭 Profiler")
                except Exception:
                    pass

    return close


def should_enable_live_profiler(args):
    """只有显式要求 profile / NVTX / 指定 tb_logdir 时才挂载 profiler 回调。"""
    return bool(getattr(args, 'torch_profile', False) or getattr(args, 'nvtx_batch', False) or str(getattr(args, 'tb_logdir', '') or '').strip())


def install_grad_clip_callback(yolo_model, max_norm=10.0):
    """仅给自定义 rainforest 系列显式安装梯度裁剪；baseline 不挂。"""
    def _grad_clip(trainer):
        try:
            if getattr(trainer, 'model', None) is not None:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=max_norm)
        except Exception:
            pass

    add_cb = getattr(yolo_model, 'add_callback', None)
    if callable(add_cb):
        add_cb('on_train_batch_end', _grad_clip)
        print(f"[TrainHook] 已为自定义模型注册梯度裁剪回调 max_norm={max_norm}")
# 保持向后兼容的别名
attach_live_profiler_to_ultralytics = install_live_profiler_ultralytics


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║              用户配置区 — 所有调参点集中于此，勿在其他地方改参数        ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║  § 1  路径 / 数据                                                       ║
# ║  § 2  共用基础参数（所有模式公共继承）                                  ║
# ║  § 3  rainforest 两阶段训练参数          --model_type rainforest        ║
# ║  § 4  rainforest_staged 三阶段渐进解冻   --model_type rainforest_staged  ║
# ║  § 5  基线 / 对比实验参数               --model_type yolo11n/s/cbam     ║
# ║  § 6  诊断 / 消融开关                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────
# § 1  路径 / 数据
# ─────────────────────────────────────────────────────────────────────────────
raw_images_path         = './data/1/3'   # 兼容旧 XML 流程；本实验默认不使用
raw_xmls_path           = './data/1/3'   # 兼容旧 XML 流程；本实验默认不使用
raw_yolo_base           = None           # 本实验改为双数据源 DATASET_SOURCES，不再直接使用单一 raw_yolo_base
save_root_path          = 'dataset_split_631'
results_path            = r'D:\college\college3\monkey\yolo\result_compare_631'
results_path_rainforest = r'D:\college\college3\monkey\yolo\result_compare_631'

# 本实验：仅使用 ordered 数据源，按 closec/closen/farc/farn 严格分层均衡后做 6:3:1
ORDERED_NEWDATA_DIR = r'D:\college\college3\monkey\yolo\datanew\ordered'

# 单一实验数据版本：
# 1) ordered : 仅 ordered 数据（images/labels 下含 closec/closen/farc/farn 四组）
DATASET_SOURCES = {
    'ordered': [ORDERED_NEWDATA_DIR],
}
SPLIT_RATIOS = (0.6, 0.3, 0.1)
SPLIT_SEED = 42
EXPERIMENT_ONE_SPLIT_ONLY = True  # 本实验固定每个数据版本仅构建 1 份划分
ORDERED_GROUPS = ('closec', 'closen', 'farc', 'farn')
STRICT_GROUP_BALANCE = True  # 严格分层均衡：四组在 train/val/test 数量完全一致

k_folds  = 1  # 兼容旧变量名；本实验不再做 K 折，而是每个数据源 1 份 6:3:1 划分
classes  = ['海南长臂猿']  # 必须与 YOLO 标签类 id 一致

# 负样本采样：本实验直接使用数据源中的全部图片与标签，不再额外采样
negative_sample_ratio = 0
negative_images_path  = None

# ─────────────────────────────────────────────────────────────────────────────
# § 2  共用基础参数（所有模式公共继承，各模式专属参数在对应 § 中覆盖）
# ─────────────────────────────────────────────────────────────────────────────
CFG_BASE = dict(
    # — 输入 ——————————————————————————————————————————
    imgsz   = 800,      # 默认 640；推荐 Phase2 用 --imgsz 800 --batch 4 减少小目标特征丢失
    batch   = 4,        # 显存 OOM 时改 4
    workers = 4,        # ★ 从 0→4：最大免费加速（0=CPU串行加载，GPU空等；4=并行预取）
    device  = 0,
    cache   = 'disk',   # ★ True→disk：离线增强后数据量 6x（~30k 张），全缓 RAM 需 30GB；disk 缓存无 OOM 风险

    # — 优化器 ————————————————————————————————————————
    optimizer    = 'AdamW',
    cos_lr       = True,
    weight_decay = 0.001,
    dropout      = 0.1,

    # — 几何增强（通用，已精简换速度） ————————————————————
    flipud      = 0.1,
    fliplr      = 0.5,
    degrees     = 0.0,      # ★ 关闭旋转（计算重 + 小目标旋转后标注不精确）
    translate   = 0.05,
    scale       = 0.4,
    perspective = 0.0,      # ★ 关闭透视变换（雨林相机固定角度，不需要）
    multi_scale = 0.3,

    # — 色彩增强（保留，计算轻且对泛化重要）————————————————
    hsv_h = 0.015,
    hsv_s = 0.8,
    hsv_v = 0.6,

    # — 其他 ——————————————————————————————————————————
    auto_augment = None,
    erasing      = 0.0,     # ★ 关闭 random erasing（离线增强已覆盖遮挡模拟）
    plots        = True,
    deterministic= False,
)

# ─────────────────────────────────────────────────────────────────────────────
# § 3  rainforest 两阶段训练参数
#      用法：python train_k.py --model_type rainforest [--single_fold]
#
#      Phase 1：冻结纯继承层（Conv / C2PSA），让新模块（AdaConv/MSCB/PAM/RST）先在
#               稳定特征上收敛；弱增强防止梯度震荡。
#      Phase 2：全解冻 + 精修增强；warmup 缩短（已有好初值）。
#      finetune：可选，主训练结束后无强增强精修，专攻 mAP50-95；0=关闭。
# ─────────────────────────────────────────────────────────────────────────────
CFG_RAINFOREST = dict(

    # ── Phase 1：冻结阶段 ─────────────────────────────────────────────────
    phase1 = dict(
        epochs        = 100,
        lr0           = 0.0005,
        lrf           = 0.01,       # 终端 lr = lr0 × lrf = 5e-6（已实验验证，勿降低）
        warmup_epochs = 15.0,       # 从头训练需长热身，避免早期梯度爆炸
        patience      = 80,
        amp           = False,      # 自定义层 FP32，防 Half() 报错

        # 损失权重（WIoU 量纲下经验平衡点；dfl 勿低于 2.0，否则 box_loss 翻倍）
        box = 9.5,
        dfl = 2.5,
        cls = 0.05,
        iou = 0.65,                 # 验证 NMS IoU

        # 增强（Phase1 弱增强，新模块先收敛）
        mosaic      = 0.3,
        close_mosaic= 10,
        mixup       = 0.0,
        copy_paste  = 0.0,
        dropout     = 0.0,          # Phase1 不 dropout，避免冻层梯度消失
        multi_scale = 0,
    ),

    # ── Phase 2：全解冻精修阶段 ───────────────────────────────────────────
    # 未列出的参数（lr0/lrf/patience/box/dfl/cls/iou）继承自 phase1
    phase2 = dict(
        epochs        = 300,
        warmup_epochs = 3.0,        # 已有好初值，无需长热身
        amp           = False,

        # 增强（适度收紧，DFL 精修）
        mosaic      = 0.2,
        close_mosaic= 60,           # ★ 精修期延长至 60ep，无 mosaic 下 DFL 坐标分布收得更准
        multi_scale = 0.3,          # ★ Phase2 开启多尺度训练（448~832），弥补 resize 特征丢失
        # 其余增强参数继承 CFG_BASE
    ),

    # ── 可选：精修阶段（主训练后追加，专攻 mAP50-95）────────────────────
    # 启用：python train_k.py --model_type rainforest --finetune_epochs 60
    # 或直接在此把 finetune_epochs 改为非 0 值
    finetune = dict(
        finetune_epochs = 0,        # 0=关闭；建议从 30~80 开始试，预期 +2~4pt
        lr0_scale       = 0.1,      # 精修 lr = phase1.lr0 × lr0_scale
        mosaic      = 0.0,
        mixup       = 0.0,
        copy_paste  = 0.0,
        erasing     = 0.0,
        multi_scale = 0,
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# § 3b  rainforest_n_lite / n_lite+ 两阶段参数
#      Phase 1 缩短（30~40 ep），避免冻结 MSCB+PAM 太久、Phase2 前半段“恢复”过慢
#      Phase 2 加长（220~260 ep），close_mosaic 适度提前（30~40）
# ─────────────────────────────────────────────────────────────────────────────
CFG_RAINFOREST_N_LITE = dict(
    phase1 = dict(
        epochs        = 25,
        lr0           = 0.0005,
        lrf           = 0.01,
        warmup_epochs = 5.0,
        patience      = 20,
        amp           = False,
        box           = 7.5,
        dfl           = 1.5,
        cls           = 0.5,
        iou           = 0.65,
        mosaic        = 0.2,
        close_mosaic  = 5,
        mixup         = 0.0,
        copy_paste    = 0.0,
        dropout       = 0.0,
        multi_scale   = 0.0,
    ),
    phase2 = dict(
        epochs        = 180,
        warmup_epochs = 2.0,
        patience      = 50,
        amp           = True,
        mosaic        = 0.15,
        close_mosaic  = 50,
        mixup         = 0.0,
        copy_paste    = 0.0,
        dropout       = 0.05,
        multi_scale   = 0.0,
    ),
)
N_LITE_FREEZE_EP   = CFG_RAINFOREST_N_LITE['phase1']['epochs']
N_LITE_UNFREEZE_EP = CFG_RAINFOREST_N_LITE['phase2']['epochs']

# ─────────────────────────────────────────────────────────────────────────────
# § 4  rainforest_staged 三阶段渐进解冻参数
#      用法：python train_k.py --model_type rainforest_staged [--phases 1 2 3] [--resume_phase N]
#
#      Phase 1 (120ep)：冻结 MSCB/PAM/RST/ST，训练 Backbone + AdaConv + Head
#      Phase 2 (120ep)：解冻 MSCB + PAM，仍冻结 RST/ST
#      Phase 3 (160ep)：全部解冻
# ─────────────────────────────────────────────────────────────────────────────
CFG_STAGED = dict(

    # 三阶段共用参数（各 phase 可覆盖）
    shared = dict(
        imgsz=640, batch=8, workers=4, device=0, cache=True,
        amp=False, dropout=0.1, patience=50,
        auto_augment=None, translate=0.05, scale=0.4,
        erasing=0.0, flipud=0.1, fliplr=0.5, degrees=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        perspective=0.0, iou=0.65,
    ),

    phase1 = dict(
        epochs               = 120,
        freeze_module_types  = ['MSCB', 'PAM', 'RST', 'ST'],
        lr0=0.002,  lrf=0.05, warmup_epochs=5.0,
        optimizer='AdamW',   weight_decay=0.001, cos_lr=True,
        mosaic=0.9, close_mosaic=20, mixup=0.05, copy_paste=0.1,
        box=7.5, cls=0.5, dfl=1.5,
    ),

    phase2 = dict(
        epochs               = 120,
        freeze_module_types  = ['RST', 'ST'],
        lr0=0.0008, lrf=0.05, warmup_epochs=3.0,
        optimizer='AdamW',   weight_decay=0.0008, cos_lr=True,
        mosaic=0.7, close_mosaic=25, mixup=0.0, copy_paste=0.05,
        box=7.5, cls=0.5, dfl=1.5,
    ),

    phase3 = dict(
        epochs               = 160,
        freeze_module_types  = [],
        lr0=0.0003, lrf=0.01, warmup_epochs=2.0,
        optimizer='AdamW',   weight_decay=0.0005, cos_lr=True,
        mosaic=0.5, close_mosaic=40, mixup=0.0, copy_paste=0.0,
        box=7.5, cls=0.5, dfl=1.5,
        multi_scale=0.3,     # ★ 全解冻阶段开启多尺度
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# § 5  基线 / 对比实验参数
#      用法：python train_k.py --model_type yolo11n / yolo11s / cbam / rainforest_arch
#
#  • CFG_BASELINE：yolo11n / cbam / rainforest_arch 等共用（当前仍继承 CFG_BASE 的 optimizer/dropout）
#  • CFG_YOLO11S_BASELINE：yolo11s 专用「原生 baseline 配方」，不继承 rainforest 特制超参，
#    用于拿到可信的 yolo11s 基线，再做 rainforest 对照。
# ─────────────────────────────────────────────────────────────────────────────
CFG_BASELINE = dict(
    epochs        = 400,
    lr0           = 0.01,
    lrf           = 0.01,
    warmup_epochs = 3.0,
    patience      = 50,
    amp           = True,

    box = 7.5,
    dfl = 1.5,
    cls = 0.5,
    iou = 0.65,

    mosaic      = 1.0,
    close_mosaic= 10,
    mixup       = 0.0,
    copy_paste  = 0.1,
    multi_scale = 0.0,
)

# yolo11s 专用：原生 YOLO 配方，不沾 rainforest 的优化器/学习率/损失权重/dropout
CFG_YOLO11S_BASELINE = dict(
    optimizer    = 'auto',   # 或 'SGD'，由 Ultralytics 默认选
    lr0          = 0.001,    # AdamW 下勿用 0.01，易发散
    lrf          = 0.01,
    warmup_epochs= 3.0,
    patience     = 50,
    amp          = True,
    dropout      = 0.0,       # baseline 不用 dropout

    box          = 7.5,       # 原生默认
    dfl          = 1.5,
    cls          = 0.5,
    iou          = 0.65,

    epochs       = 400,
    mosaic       = 1.0,
    close_mosaic = 10,
    mixup        = 0.0,
    copy_paste   = 0.1,
    multi_scale  = 0.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# § 6  诊断 / 消融开关
#      均为 False 时走正常流程；True 时自动覆盖当前模式对应参数
# ─────────────────────────────────────────────────────────────────────────────
CFG_DEBUG = dict(
    # NaN 安全模式：关 AMP、降 lr、弱增强，用于排查梯度 NaN
    use_nan_safe    = False,
    nan_safe_patch  = dict(
        amp=False, lr0=0.003, warmup_epochs=10.0,
        mosaic=0.5, mixup=0.0, copy_paste=0.0, close_mosaic=30,
        erasing=0.0, translate=0.04, scale=0.3, auto_augment=None,
    ),
    # 抗误检模式：弱结构增强 + 强光照扰动，用于降低视频误检率
    use_antifp_aug  = False,
    antifp_patch    = dict(
        mosaic=0.15, mixup=0.0, copy_paste=0.0, close_mosaic=30,
        erasing=0.08, translate=0.04, scale=0.3, auto_augment=None,
    ),
)

# ╚══════════════════════════════════════════════════════════════════════════╝
# 以下为内部计算别名，勿直接修改
# ╔══════════════════════════════════════════════════════════════════════════╝

# rainforest 主训练参数（phase1 作基础，debug 覆盖在上）
_rf_p1 = CFG_RAINFOREST['phase1']
TRAIN_PARAMS = {**CFG_BASE, **_rf_p1}
# n-lite 两阶段用 phase1 作 active_params 基础
TRAIN_PARAMS_N_LITE = {**CFG_BASE, **CFG_RAINFOREST_N_LITE['phase1']}
if CFG_DEBUG['use_nan_safe']:
    TRAIN_PARAMS.update(CFG_DEBUG['nan_safe_patch'])
if CFG_DEBUG['use_antifp_aug']:
    TRAIN_PARAMS.update(CFG_DEBUG['antifp_patch'])

# ─────────────────────────────────────────────────────────────────────────────
# § 2.5  各模型独立训练配置（互不继承，避免 baseline 被 rainforest 公共参数污染）
#      说明：
#      - yolo11n / yolo11s / cbam / rainforest_arch 走“纯 baseline / 纯对照”参数
#      - rainforest 系列继续使用各自专属实验参数
#      - all 模式本身已通过 subprocess.run 为每个模型启动独立 Python 进程，这里再把参数源也彻底拆开
# ─────────────────────────────────────────────────────────────────────────────
def _cfg_clone(d):
    return copy.deepcopy(d)


# ─────────────────────────────────────────────────────────────────────────────
# § 2.5  论文高保真复现配置（尽可能严格对齐 Rainforest-YOLO 论文可见信息）
#      说明：
#      - 论文公开可确定的硬约束：YOLOv8s 基线 + Rainforest-YOLO(v8s family)
#      - 论文未公开完整训练超参；以下训练配置采用同一套 shared recipe，确保基线与改进模型公平对照
#      - 不再混用 YOLO11 / staged / n-lite 等工程扩展分支
# ─────────────────────────────────────────────────────────────────────────────
PAPER_SHARED_TRAIN_CFG = dict(
    imgsz=640,
    batch=8,
    nbs=64,
    workers=4,
    device=0,
    cache='disk',
    optimizer='auto',
    cos_lr=False,
    weight_decay=0.0005,
    dropout=0.0,
    flipud=0.0,
    fliplr=0.5,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    perspective=0.0,
    multi_scale=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    auto_augment=None,
    erasing=0.0,
    plots=True,
    deterministic=False,
    epochs=300,
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3.0,
    patience=50,
    amp=True,
    box=7.5,
    dfl=1.5,
    cls=0.5,
    iou=0.65,
    mosaic=1.0,
    close_mosaic=10,
    mixup=0.0,
    copy_paste=0.1,
)

CFG_YOLOV8S_PAPER_BASELINE = _cfg_clone(PAPER_SHARED_TRAIN_CFG)
CFG_RAINFOREST_V8S_PAPER = _cfg_clone(PAPER_SHARED_TRAIN_CFG)
CFG_RAINFOREST_V8S_PAPER['amp'] = False  # RST/LayerNorm 在验证期与 AMP 半精度存在 dtype 冲突，论文版暂用 FP32

# rainforest_paper_v8s 三阶段受控训练（避免与离线增强分布冲突）
# 阶段总 epoch = 100 + 180 + 40 = 320
CFG_RAINFOREST_V8S_PAPER_PHASED = dict(
    shared=dict(
        imgsz=800,
        batch=4,
        nbs=8,
        device=0,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        amp=False,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment=None,
        erasing=0.0,
    ),
    phase1=dict(  # 结构收敛
        epochs=100,
        warmup_epochs=5.0,
        patience=80,
        mosaic=0.2,
        close_mosaic=10,
        multi_scale=0.0,
    ),
    phase2=dict(  # 泛化学习
        epochs=180,
        warmup_epochs=3.0,
        patience=120,
        mosaic=0.1,
        close_mosaic=30,
        multi_scale=0.3,
    ),
    phase3=dict(  # 精修
        epochs=40,
        warmup_epochs=1.0,
        patience=40,
        mosaic=0.0,
        close_mosaic=0,
        multi_scale=0.0,
        lr0_scale=0.1,
    ),
)

# YOLOv8s-MonkeyPlus：P2 四尺度 + 少量 MSCB + PAM/ResPAM；无 RST / 无 AdaConv 全替换
# 训练脚本内二阶段：640×300ep + 800×50ep 精修（见主循环）
CFG_MONKEYPLUS_PHASED = dict(
    shared=dict(
        optimizer='auto',
        lrf=0.01,
        amp=True,
        mixup=0.0,
        auto_augment=None,
        erasing=0.0,
    ),
    phase1=dict(
        epochs=300,
        imgsz=640,
        batch=8,
        nbs=64,
        lr0=0.001,
        warmup_epochs=3.0,
        patience=50,
        mosaic=1.0,
        close_mosaic=30,
        multi_scale=0.0,
        copy_paste=0.1,
    ),
    phase2=dict(
        epochs=50,
        imgsz=800,
        batch=4,
        nbs=64,
        lr0_scale=0.1,
        warmup_epochs=1.0,
        patience=30,
        mosaic=0.0,
        close_mosaic=0,
        multi_scale=0.0,
        copy_paste=0.0,
    ),
)

CFG_YOLOV8S_ABLATION_SHARED = _cfg_clone(PAPER_SHARED_TRAIN_CFG)

# 消融实验：与 PAPER_SHARED_TRAIN_CFG 相同超参；含 RST 的 YAML 在训练入口另设 amp=False
ABLATION_MODEL_METADATA = {
    'yolov8s_ablate_baseline': {
        'yaml': 'cfg/yolov8s-ablate-baseline.yaml',
        'adaconv': False, 'mscb': False, 'pam': False, 'rst_pos': 'none',
    },
    'yolov8s_ablate_adaconv_only': {
        'yaml': 'cfg/yolov8s-ablate-adaconv-only.yaml',
        'adaconv': True, 'mscb': False, 'pam': False, 'rst_pos': 'none',
    },
    'yolov8s_ablate_mscb_only': {
        'yaml': 'cfg/yolov8s-ablate-mscb-only.yaml',
        'adaconv': False, 'mscb': True, 'pam': False, 'rst_pos': 'none',
    },
    'yolov8s_ablate_pam_only': {
        'yaml': 'cfg/yolov8s-ablate-pam-only.yaml',
        'adaconv': False, 'mscb': False, 'pam': True, 'rst_pos': 'none',
    },
    'yolov8s_ablate_rst_only': {
        'yaml': 'cfg/yolov8s-ablate-rst-only.yaml',
        'adaconv': False, 'mscb': False, 'pam': False, 'rst_pos': 'p3p4p5',
    },
    'yolov8s_ablate_mscb_pam': {
        'yaml': 'cfg/yolov8s-ablate-mscb-pam.yaml',
        'adaconv': False, 'mscb': True, 'pam': True, 'rst_pos': 'none',
    },
    'yolov8s_ablate_no_rst': {
        'yaml': 'cfg/yolov8s-ablate-no-rst.yaml',
        'adaconv': True, 'mscb': True, 'pam': True, 'rst_pos': 'none',
    },
    'yolov8s_ablate_rst_p5_only': {
        'yaml': 'cfg/yolov8s-ablate-rst-p5-only.yaml',
        'adaconv': True, 'mscb': True, 'pam': True, 'rst_pos': 'p5',
    },
    'yolov8s_ablate_rst_p4p5': {
        'yaml': 'cfg/yolov8s-ablate-rst-p4p5.yaml',
        'adaconv': True, 'mscb': True, 'pam': True, 'rst_pos': 'p4p5',
    },
    'yolov8s_ablate_full': {
        'yaml': 'cfg/yolov8s-ablate-full.yaml',
        'adaconv': True, 'mscb': True, 'pam': True, 'rst_pos': 'p3p4p5',
    },
}

YOLOV8S_ABLATION_TYPES = frozenset(ABLATION_MODEL_METADATA.keys())

MONKEYPLUS_MODEL_METADATA = {
    'yolov8s_monkeyplus_p2': {
        'yaml': 'cfg/yolov8s-monkeyplus-p2.yaml',
    },
    'yolov8s_monkeyplus_p2_respam': {
        'yaml': 'cfg/yolov8s-monkeyplus-p2-respam.yaml',
    },
}
YOLOV8S_MONKEYPLUS_TYPES = frozenset(MONKEYPLUS_MODEL_METADATA.keys())

# YOLOv8s.pt + P2 路线（无 MSCB / 无多路 PAM / 无 RST）：单阶段主训 + 可选 800 精修
# 实验顺序见文档：p2_only → cm30 → ft800 → p2_wide → p2_wide_respam
P2_TRACK_MODEL_METADATA = {
    'yolov8s_p2_only': {'yaml': 'cfg/yolov8s-p2-only.yaml'},
    'yolov8s_p2_only_cm30': {'yaml': 'cfg/yolov8s-p2-only.yaml'},
    'yolov8s_p2_only_cm30_ft800': {'yaml': 'cfg/yolov8s-p2-only.yaml'},
    'yolov8s_p2_wide': {'yaml': 'cfg/yolov8s-p2-wide.yaml'},
    'yolov8s_p2_wide_respam': {'yaml': 'cfg/yolov8s-p2-wide-respam.yaml'},
    'yolov8s_p2_respam_p2only': {'yaml': 'cfg/yolov8s-p2-respam-p2only.yaml'},
    'yolov8s_p2_respam_p2only_gql_nwd': {'yaml': 'cfg/yolov8s-p2-respam-p2only.yaml'},
    'yolov8s_p2_respam_decoupled_p2head': {'yaml': 'cfg/yolov8s-p2-respam-decoupled-p2head.yaml'},
    'yolov8s_p2_respam_hfres': {'yaml': 'cfg/yolov8s-p2-respam-hfres.yaml'},
    'yolov8s_p2_respam_hfres_ft800': {'yaml': 'cfg/yolov8s-p2-respam-hfres.yaml'},
    'yolov8s_p2_respam_sgf': {'yaml': 'cfg/yolov8s-p2-respam-sgf.yaml'},
    'yolov8s_p2_respam_sgf_ft800': {'yaml': 'cfg/yolov8s-p2-respam-sgf.yaml'},
    'yolov8s_p2_wide_respam_sgf': {'yaml': 'cfg/yolov8s-p2-wide-respam-sgf.yaml'},
    'yolov8s_p2_wide_respam_sgf_ft800': {'yaml': 'cfg/yolov8s-p2-wide-respam-sgf.yaml'},
    'yolov8s_p2_bifpn_lite': {'yaml': 'cfg/yolov8s-p2-bifpn-lite.yaml'},
    'yolov8s_p2_hfres': {'yaml': 'cfg/yolov8s-p2-hfres.yaml'},
    'yolov8s_p2_sgf': {'yaml': 'cfg/yolov8s-p2-sgf.yaml'},
    'yolov8s_p2_hfres_sgf': {'yaml': 'cfg/yolov8s-p2-hfres-sgf.yaml'},
    'yolov8s_p2_hfres_sgf_ft800': {'yaml': 'cfg/yolov8s-p2-hfres-sgf.yaml'},
    # GibbonGuard：P2 降级为细节注入分支（保留 ResPAM@P2），最终 Detect 仅 P3/P4/P5
    'yolov8s_p2_aux_p3main': {'yaml': 'cfg/yolov8s-p2-aux-p3main.yaml'},
    # GibbonGuard v2：在 v1 基础上加 training-only 的 P2 辅助监督头（AuxDetect），推理仍 3 头
    'yolov8s_p2_auxsup_p3main': {'yaml': 'cfg/yolov8s-p2-auxsup-p3main.yaml'},
}
YOLOV8S_P2_TRACK_TYPES = frozenset(P2_TRACK_MODEL_METADATA.keys())
GIBBON_QUALITY_LOSS_TYPES = frozenset({'yolov8s_p2_respam_p2only_gql_nwd'})

YOLOV8S_PT_YAML_MODEL_METADATA = {**MONKEYPLUS_MODEL_METADATA, **P2_TRACK_MODEL_METADATA}
YOLOV8S_PT_YAML_TYPES = frozenset(YOLOV8S_PT_YAML_MODEL_METADATA.keys())

# P2 路线 800 精修（lr0 = 主训 lr0 × lr0_scale，默认 0.001×0.1=1e-4）
CFG_P2_FINETUNE = dict(
    finetune_epochs=50,
    lr0_scale=0.1,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.0,
    multi_scale=0.0,
    close_mosaic=0,
)

# 各 model_type 默认是否跑精修（可用 --finetune_epochs 覆盖）
P2_MODEL_DEFAULT_FINETUNE_EPOCHS = {
    'yolov8s_p2_only': 0,
    'yolov8s_p2_only_cm30': 0,
    'yolov8s_p2_only_cm30_ft800': 50,
    'yolov8s_p2_wide': 0,
    'yolov8s_p2_wide_respam': 0,
    'yolov8s_p2_respam_p2only': 0,
    'yolov8s_p2_respam_p2only_gql_nwd': 0,
    'yolov8s_p2_respam_decoupled_p2head': 0,
    'yolov8s_p2_respam_hfres': 0,
    'yolov8s_p2_respam_hfres_ft800': 50,
    'yolov8s_p2_respam_sgf': 0,
    'yolov8s_p2_respam_sgf_ft800': 50,
    'yolov8s_p2_wide_respam_sgf': 0,
    'yolov8s_p2_wide_respam_sgf_ft800': 50,
    'yolov8s_p2_bifpn_lite': 0,
    'yolov8s_p2_hfres': 0,
    'yolov8s_p2_sgf': 0,
    'yolov8s_p2_hfres_sgf': 0,
    'yolov8s_p2_hfres_sgf_ft800': 50,
    'yolov8s_p2_aux_p3main': 0,
    'yolov8s_p2_auxsup_p3main': 0,
}


def is_yolov8s_pt_yaml_model(model_type: str) -> bool:
    """是否使用 yolov8s.pt 预训练 + 本地 YAML 构建（含 MonkeyPlus 与 P2-only）。"""
    return model_type in YOLOV8S_PT_YAML_TYPES


def is_monkeyplus_phased_model(model_type: str) -> bool:
    """MonkeyPlus 二阶段（640+800），与 P2-only 单阶段路径区分。"""
    return model_type in YOLOV8S_MONKEYPLUS_TYPES


def is_p2_track_model(model_type: str) -> bool:
    return model_type in YOLOV8S_P2_TRACK_TYPES


def is_p2_only_clean_model(model_type: str) -> bool:
    """P2 路线（含 wide / cm30 / ft800）；不含 MonkeyPlus。"""
    return is_p2_track_model(model_type)


def is_monkeyplus_model(model_type: str) -> bool:
    """兼容旧名：等同于 is_yolov8s_pt_yaml_model。"""
    return is_yolov8s_pt_yaml_model(model_type)

MODEL_PARAM_PRESETS = {
    'yolov8s': CFG_YOLOV8S_PAPER_BASELINE,
    'rainforest_paper_v8s': CFG_RAINFOREST_V8S_PAPER,
}
MODEL_PARAM_PRESETS.update({k: _cfg_clone(CFG_YOLOV8S_ABLATION_SHARED) for k in YOLOV8S_ABLATION_TYPES})
MODEL_PARAM_PRESETS.update({k: _cfg_clone(CFG_YOLOV8S_ABLATION_SHARED) for k in YOLOV8S_MONKEYPLUS_TYPES})


def _p2_train_preset(close_mosaic: int) -> dict:
    p = _cfg_clone(CFG_YOLOV8S_ABLATION_SHARED)
    p['close_mosaic'] = close_mosaic
    return p


# close_mosaic：p2_only=10（对照 ablation）；cm30 系列=30
MODEL_PARAM_PRESETS['yolov8s_p2_only'] = _p2_train_preset(10)
MODEL_PARAM_PRESETS['yolov8s_p2_only_cm30'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_only_cm30_ft800'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_wide'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_wide_respam'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_respam_p2only'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_respam_p2only_gql_nwd'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_respam_decoupled_p2head'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_aux_p3main'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_auxsup_p3main'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_respam_hfres'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_respam_hfres_ft800'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_respam_sgf'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_respam_sgf_ft800'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_wide_respam_sgf'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_wide_respam_sgf_ft800'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_bifpn_lite'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_hfres'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_sgf'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_hfres_sgf'] = _p2_train_preset(30)
MODEL_PARAM_PRESETS['yolov8s_p2_hfres_sgf_ft800'] = _p2_train_preset(30)


def is_yolov8s_ablation_model(model_type: str) -> bool:
    return model_type in YOLOV8S_ABLATION_TYPES


def ablation_model_has_rst(model_type: str) -> bool:
    meta = ABLATION_MODEL_METADATA.get(model_type) or {}
    return meta.get('rst_pos', 'none') != 'none'


def is_pure_baseline_model(model_type: str) -> bool:
    return model_type in ('yolov8s', 'yolov8s_ablate_baseline')


def is_custom_rainforest_model(model_type: str) -> bool:
    # 论文高保真复现模式默认不额外挂梯度裁剪；返回 False 保持训练路径更接近标准 Ultralytics
    return False


def resolve_model_active_params(model_type: str):
    if model_type not in MODEL_PARAM_PRESETS:
        raise KeyError(f'未知 model_type: {model_type}')
    return _cfg_clone(MODEL_PARAM_PRESETS[model_type]), ('pure_baseline' if is_pure_baseline_model(model_type) else 'paper_faithful')

# staged 内部引用别名（保持 run_staged_training 代码不变）
PHASE_CONFIGS = {
    1: CFG_STAGED['phase1'],
    2: CFG_STAGED['phase2'],
    3: CFG_STAGED['phase3'],
}
STAGED_SHARED_PARAMS = CFG_STAGED['shared']

# rainforest 两阶段 epoch 数（逻辑代码引用）
RAINFOREST_FREEZE_EP   = CFG_RAINFOREST['phase1']['epochs']
RAINFOREST_UNFREEZE_EP = CFG_RAINFOREST['phase2']['epochs']

# ==================== [2. 定义并注入 CBAM 模块（仅用于 --model_type cbam，非论文架构）] ====================
# 论文 PAM 为 CAM 与 SAM 并行后求和，见 ultralytics_rainforest.PAM；此处 CBAM 为串行 CA→SA 再乘回 x。
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(torch.nn.functional.adaptive_avg_pool2d(x, 1))
        max_out = self.fc(torch.nn.functional.adaptive_max_pool2d(x, 1))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))

class CBAM(nn.Module):
    def __init__(self, c1, k=7):
        super().__init__()
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention(k)
    def forward(self, x):
        x = self.ca(x) * x
        return self.sa(x) * x

setattr(tasks, 'CBAM', CBAM)

# ==================== [3. 数据处理工具函数] ====================
def convert(size, box):
    """box: (xmin, xmax, ymin, ymax) 像素坐标，已做 clamp/swap。返回 YOLO 归一化 (cx, cy, w, h)。"""
    dw, dh = 1. / size[0], 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w, h = box[1] - box[0], box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def _sanitize_box(xmin, xmax, ymin, ymax, img_w, img_h):
    """交换逆序、clamp 到图像内、过滤无效框。返回 (xmin,xmax,ymin,ymax) 或 None（应跳过）。"""
    xmin, xmax = sorted([xmin, xmax])
    ymin, ymax = sorted([ymin, ymax])
    xmin = max(0.0, min(float(xmin), float(img_w)))
    xmax = max(0.0, min(float(xmax), float(img_w)))
    ymin = max(0.0, min(float(ymin), float(img_h)))
    ymax = max(0.0, min(float(ymax), float(img_h)))
    if (xmax - xmin) < 1 or (ymax - ymin) < 1:
        return None
    return (xmin, xmax, ymin, ymax)

def process_xml(xml_path, txt_save_path):
    """
    解析 XML 并生成 YOLO 格式标签文件。对框做 clamp/swap/skip，避免负宽高、越界、极小框导致 NaN。

    Args:
        xml_path: XML 文件路径
        txt_save_path: 输出 txt 文件路径（如果为 None，仅返回目标数量，不写入文件）

    Returns:
        found_objects: 找到的目标数量（0 表示无目标，即负样本）
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        w, h = int(size.find('width').text), int(size.find('height').text)
        found_objects = 0

        if txt_save_path is None:
            for obj in root.iter('object'):
                cls_name = obj.find('name').text.strip()
                if cls_name not in classes:
                    continue
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                if _sanitize_box(b[0], b[1], b[2], b[3], w, h) is None:
                    continue
                found_objects += 1
            return found_objects

        with open(txt_save_path, 'w', encoding='utf-8') as out_file:
            for obj in root.iter('object'):
                cls_name = obj.find('name').text.strip()
                if cls_name not in classes:
                    continue
                cls_id = classes.index(cls_name)
                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymin = float(xmlbox.find('ymin').text)
                ymax = float(xmlbox.find('ymax').text)
                san = _sanitize_box(xmin, xmax, ymin, ymax, w, h)
                if san is None:
                    continue
                bb = convert((w, h), san)
                if not all(0.0 <= v <= 1.0 and math.isfinite(v) for v in bb):
                    continue
                out_file.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")
                found_objects += 1
        return found_objects
    except Exception:
        return 0

def export_xml_to_yolo(output_base='data_yolo'):
    """
    将当前 raw_images_path + raw_xmls_path（XML）导出为 YOLO 格式到 output_base/images 与 output_base/labels。
    之后可对 output_base 运行 augment_rainforest_offline.py，再设 raw_yolo_base 指向增强目录做 K 折。
    """
    img_out = os.path.join(output_base, 'images')
    lbl_out = os.path.join(output_base, 'labels')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    image_files = [f for f in os.listdir(raw_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n_pos, n_neg = 0, 0
    for img_name in image_files:
        stem = os.path.splitext(img_name)[0]
        xml_path = os.path.join(raw_xmls_path, stem + '.xml')
        txt_path = os.path.join(lbl_out, stem + '.txt')
        src_img = os.path.join(raw_images_path, img_name)
        if not os.path.exists(src_img):
            continue
        if os.path.exists(xml_path):
            n_obj = process_xml(xml_path, txt_path)
            if n_obj > 0:
                n_pos += 1
            else:
                open(txt_path, 'w').close()
                n_neg += 1
        else:
            open(txt_path, 'w').close()
            n_neg += 1
        ext = os.path.splitext(img_name)[1].lower() or '.jpg'
        shutil.copy(src_img, os.path.join(img_out, stem + ext))
    print(f"XML→YOLO 导出完成: {output_base}  正样本 {n_pos} 张, 负样本 {n_neg} 张")
    return n_pos, n_neg


def check_yolo_labels(lbl_dir):
    """
    检查 YOLO 标签目录下所有 .txt：负宽高、越界、NaN/Inf、len!=5 等异常。
    返回 [(path, line_no, reason, raw_line), ...]，用于训练前自检，避免 NaN。
    """
    bad = []
    if not os.path.isdir(lbl_dir):
        return bad
    for fname in os.listdir(lbl_dir):
        if not fname.lower().endswith('.txt'):
            continue
        p = os.path.join(lbl_dir, fname)
        with open(p, 'r', encoding='utf-8') as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) != 5:
                    bad.append((p, ln, "len!=5", s))
                    continue
                try:
                    cls_str, x, y, w, h = parts
                    cls_id = int(cls_str)
                    x, y, w, h = float(x), float(y), float(w), float(h)
                except Exception:
                    bad.append((p, ln, "parse_fail", s))
                    continue
                if cls_id < 0 or cls_id >= len(classes):
                    bad.append((p, ln, f"cls_id={cls_id}_oob(max={len(classes)-1})", s))
                    continue
                if not all(math.isfinite(v) for v in (x, y, w, h)):
                    bad.append((p, ln, "nan/inf", s))
                    continue
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    bad.append((p, ln, "xy_oob", s))
                    continue
                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    bad.append((p, ln, "wh_invalid", s))
                    continue
    return bad

def _resolve_yolo_dirs(base_dir):
    """解析数据源目录，优先使用 base/images + base/labels；兼容 images/train + labels/train。"""
    img_root = os.path.join(base_dir, 'images')
    lbl_root = os.path.join(base_dir, 'labels')
    if os.path.isdir(img_root) and os.path.isdir(lbl_root):
        img_train = os.path.join(img_root, 'train')
        lbl_train = os.path.join(lbl_root, 'train')
        if os.path.isdir(img_train) and os.path.isdir(lbl_train):
            return img_train, lbl_train
        return img_root, lbl_root
    raise FileNotFoundError(f"未找到 YOLO 数据目录: {base_dir} (期望存在 images/ 和 labels/)")


def _list_image_files(img_dir):
    return sorted(f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')))

def _dataset_source_dirs(ds_spec):
    if isinstance(ds_spec, (list, tuple)):
        return list(ds_spec)
    return [ds_spec]


def _parse_and_validate_yolo_line(line):
    s = line.strip()
    if not s:
        return None, 'empty'
    parts = s.split()
    if len(parts) != 5:
        return None, 'len!=5'
    try:
        cls_str, x, y, w, h = parts
        cls_id = int(cls_str)
        x, y, w, h = float(x), float(y), float(w), float(h)
    except Exception:
        return None, 'parse_fail'
    if cls_id < 0 or cls_id >= len(classes):
        return None, f'cls_id={cls_id}_oob(max={len(classes)-1})'
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None, 'nan/inf'
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None, 'xy_oob'
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return None, 'wh_invalid'
    return f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}", None


def _sanitize_label_file(src_txt_path):
    valid_lines = []
    dropped = []
    if not os.path.exists(src_txt_path):
        return valid_lines, dropped
    try:
        with open(src_txt_path, 'r', encoding='utf-8') as f:
            for ln, line in enumerate(f, 1):
                norm_line, reason = _parse_and_validate_yolo_line(line)
                if norm_line is not None:
                    valid_lines.append(norm_line)
                else:
                    s = line.strip()
                    if s:
                        dropped.append((ln, reason, s))
    except Exception as e:
        dropped.append((0, f'read_fail:{e}', ''))
    return valid_lines, dropped


def _has_valid_yolo_label(txt_path):
    valid_lines, _ = _sanitize_label_file(txt_path)
    return len(valid_lines) > 0


def _label_signature(txt_path):
    if not os.path.exists(txt_path):
        return 'missing'
    try:
        with open(txt_path, 'rb') as f:
            data = f.read()
        return hashlib.md5(data).hexdigest()[:12]
    except Exception:
        return 'read_error'


def _experiment_fingerprint():
    parts = [
        f"classes={classes}",
        f"ratios={SPLIT_RATIOS}",
        f"seed={SPLIT_SEED}",
        "label_cleaner=v3",
        "dataset_groups=raw|enhanced|all",
        f"ordered_groups={','.join(ORDERED_GROUPS)}",
        f"strict_group_balance={int(STRICT_GROUP_BALANCE)}",
    ]
    for ds_name, ds_spec in sorted(DATASET_SOURCES.items()):
        for src_idx, base_dir in enumerate(_dataset_source_dirs(ds_spec)):
            try:
                img_dir, lbl_dir = _resolve_yolo_dirs(base_dir)
                files = _list_image_files(img_dir)
                parts.append(f"{ds_name}:src{src_idx}:{base_dir}:n={len(files)}")
                parts.extend(f"{ds_name}:src{src_idx}:{f}" for f in files)
                for f in files:
                    stem = os.path.splitext(f)[0]
                    txt = os.path.join(lbl_dir, stem + '.txt')
                    parts.append(f"{ds_name}:src{src_idx}:{stem}:label_valid={int(_has_valid_yolo_label(txt))}:sig={_label_signature(txt)}")
            except Exception as e:
                parts.append(f"{ds_name}:src{src_idx}:ERR:{base_dir}:{e}")
    raw = "\n".join(parts)
    return hashlib.md5(raw.encode('utf-8')).hexdigest()[:16]

def _copy_sample(src_img_path, src_txt_path, dst_img_dir, dst_lbl_dir, new_base_name, sanitize_labels=True, dropped_log=None):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    ext = os.path.splitext(src_img_path)[1].lower() or '.jpg'
    shutil.copy2(src_img_path, os.path.join(dst_img_dir, new_base_name + ext))
    dst_txt = os.path.join(dst_lbl_dir, new_base_name + '.txt')
    if os.path.exists(src_txt_path):
        if sanitize_labels:
            valid_lines, dropped = _sanitize_label_file(src_txt_path)
            with open(dst_txt, 'w', encoding='utf-8') as f:
                if valid_lines:
                    f.write("\n".join(valid_lines) + "\n")
            if dropped and dropped_log is not None:
                dropped_log.append({
                    'src_txt': src_txt_path,
                    'dst_txt': dst_txt,
                    'dropped': dropped,
                    'kept': len(valid_lines),
                })
        else:
            shutil.copy2(src_txt_path, dst_txt)
    else:
        open(dst_txt, 'w', encoding='utf-8').close()


def _split_indices(n, ratios):
    tr, va, te = ratios
    assert abs((tr + va + te) - 1.0) < 1e-8, f"划分比例和必须为 1，当前为 {ratios}"
    n_train = int(n * tr)
    n_val = int(n * va)
    n_test = n - n_train - n_val
    if n >= 3:
        if n_train <= 0:
            n_train = 1
        if n_val <= 0:
            n_val = 1
        if n_test <= 0:
            n_test = 1
        while n_train + n_val + n_test > n:
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        while n_train + n_val + n_test < n:
            n_train += 1
    return n_train, n_val, n_test


def _collect_samples_from_dir(base_dir, src_idx):
    """
    采样逻辑同时兼容两种目录：
    1) 常规 YOLO 结构: base/images + base/labels
    2) 分组结构: base/images/<group> + base/labels/<group>
    """
    samples = []
    grouped_samples = {g: [] for g in ORDERED_GROUPS}
    img_dir, lbl_dir = _resolve_yolo_dirs(base_dir)

    has_grouped = all(
        os.path.isdir(os.path.join(img_dir, g)) and os.path.isdir(os.path.join(lbl_dir, g))
        for g in ORDERED_GROUPS
    )

    if has_grouped:
        for g in ORDERED_GROUPS:
            g_img_dir = os.path.join(img_dir, g)
            g_lbl_dir = os.path.join(lbl_dir, g)
            image_files = _list_image_files(g_img_dir)
            for img_name in image_files:
                stem = os.path.splitext(img_name)[0]
                txt_path = os.path.join(g_lbl_dir, stem + '.txt')
                item = {
                    'img_name': img_name,
                    'img_path': os.path.join(g_img_dir, img_name),
                    'txt_path': txt_path,
                    'is_pos': _has_valid_yolo_label(txt_path),
                    'source_idx': src_idx,
                    'source_dir': base_dir,
                    'group': g,
                }
                samples.append(item)
                grouped_samples[g].append(item)
        return samples, grouped_samples, True

    image_files = _list_image_files(img_dir)
    for img_name in image_files:
        stem = os.path.splitext(img_name)[0]
        txt_path = os.path.join(lbl_dir, stem + '.txt')
        samples.append({
            'img_name': img_name,
            'img_path': os.path.join(img_dir, img_name),
            'txt_path': txt_path,
            'is_pos': _has_valid_yolo_label(txt_path),
            'source_idx': src_idx,
            'source_dir': base_dir,
            'group': None,
        })
    return samples, grouped_samples, False


def _extract_dataset_name_from_yaml(yaml_file):
    split_dir = os.path.dirname(os.path.abspath(yaml_file))
    return os.path.basename(os.path.dirname(split_dir))


def prepare_kfold_data():
    """
    兼容旧函数名：不再做 K-Fold，而是对 ordered 数据构建 1 份 6:3:1 划分。
    ordered 数据要求：
      - images/closec|closen|farc|farn
      - labels/closec|closen|farc|farn
    并对四个分组执行严格分层均衡划分（train/val/test 数量一致）。
    返回值仍为 yaml 路径列表，供后续训练循环复用。
    """
    os.makedirs(save_root_path, exist_ok=True)
    fp_file = os.path.join(save_root_path, '.fingerprint')
    cur_fp = _experiment_fingerprint()
    if os.path.isfile(fp_file):
        try:
            cached_fp = open(fp_file, 'r', encoding='utf-8').read().strip()
            if cached_fp == cur_fp:
                yaml_paths = []
                for ds_name in sorted(DATASET_SOURCES.keys()):
                    cand = os.path.join(save_root_path, ds_name, 'split_1', 'data.yaml')
                    if os.path.isfile(cand):
                        yaml_paths.append(cand)
                if len(yaml_paths) == len(DATASET_SOURCES):
                    print(f"✅ 数据划分指纹匹配，跳过重建（{save_root_path}）")
                    return yaml_paths
        except Exception:
            pass

    if os.path.exists(save_root_path):
        shutil.rmtree(save_root_path)
    os.makedirs(save_root_path, exist_ok=True)

    yaml_paths = []
    label_clean_reports = []
    for ds_name, ds_spec in sorted(DATASET_SOURCES.items()):
        source_dirs = _dataset_source_dirs(ds_spec)
        samples = []
        pos_count = 0
        neg_count = 0
        source_summaries = []
        grouped_samples = {g: [] for g in ORDERED_GROUPS}
        has_grouped_layout = False

        for src_idx, base_dir in enumerate(source_dirs):
            src_samples, src_grouped, src_has_grouped = _collect_samples_from_dir(base_dir, src_idx)
            if not src_samples:
                raise RuntimeError(f"数据源为空: {base_dir}")
            has_grouped_layout = has_grouped_layout or src_has_grouped
            samples.extend(src_samples)
            for g in ORDERED_GROUPS:
                grouped_samples[g].extend(src_grouped[g])

            src_pos = sum(1 for x in src_samples if x['is_pos'])
            src_neg = len(src_samples) - src_pos
            pos_count += src_pos
            neg_count += src_neg
            source_summaries.append((base_dir, len(src_samples), src_pos, src_neg, src_has_grouped))

        print(f"📊 数据版本 {ds_name}: 总样本 {len(samples)}，正样本 {pos_count}，负样本 {neg_count}")
        for base_dir, n_img, src_pos, src_neg, src_has_grouped in source_summaries:
            layout_note = "分组结构" if src_has_grouped else "普通结构"
            print(f"   ├─ 来源 {base_dir}: 样本 {n_img}，正样本 {src_pos}，负样本 {src_neg}，布局={layout_note}")

        rng = random.Random(SPLIT_SEED)
        if has_grouped_layout and STRICT_GROUP_BALANCE:
            group_sizes = {g: len(grouped_samples[g]) for g in ORDERED_GROUPS}
            min_group_n = min(group_sizes.values()) if group_sizes else 0
            if min_group_n <= 0:
                raise RuntimeError(f"分组均衡划分失败：至少一个分组为空，统计={group_sizes}")
            n_train_g, n_val_g, n_test_g = _split_indices(min_group_n, SPLIT_RATIOS)
            print(f"   分层均衡(严格): 各组原始数量={group_sizes}，按最小组截断={min_group_n}")
            print(f"   单组 6:3:1 -> train={n_train_g} val={n_val_g} test={n_test_g}")

            train_samples, val_samples, test_samples = [], [], []
            for g in ORDERED_GROUPS:
                arr = list(grouped_samples[g])
                rng.shuffle(arr)
                arr = arr[:min_group_n]
                train_samples.extend(arr[:n_train_g])
                val_samples.extend(arr[n_train_g:n_train_g + n_val_g])
                test_samples.extend(arr[n_train_g + n_val_g:n_train_g + n_val_g + n_test_g])

            rng.shuffle(train_samples)
            rng.shuffle(val_samples)
            rng.shuffle(test_samples)
            print(
                f"   严格均衡结果 -> "
                f"train={len(train_samples)}({n_train_g}x{len(ORDERED_GROUPS)}) "
                f"val={len(val_samples)}({n_val_g}x{len(ORDERED_GROUPS)}) "
                f"test={len(test_samples)}({n_test_g}x{len(ORDERED_GROUPS)})"
            )
        else:
            rng.shuffle(samples)
            n_train, n_val, n_test = _split_indices(len(samples), SPLIT_RATIOS)
            train_samples = samples[:n_train]
            val_samples = samples[n_train:n_train + n_val]
            test_samples = samples[n_train + n_val:]
            print(f"   6:3:1 划分 -> train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

        split_root = os.path.join(save_root_path, ds_name, 'split_1')
        ds_dropped_log = []
        for subset, subset_samples in (('train', train_samples), ('val', val_samples), ('test', test_samples)):
            subset_img_dir = os.path.join(split_root, subset, 'images')
            subset_lbl_dir = os.path.join(split_root, subset, 'labels')
            for idx, item in enumerate(tqdm(subset_samples, desc=f"构建 {ds_name}/{subset}")):
                new_base_name = f"{idx:06d}"
                _copy_sample(item['img_path'], item['txt_path'], subset_img_dir, subset_lbl_dir, new_base_name, sanitize_labels=True, dropped_log=ds_dropped_log)

        if ds_dropped_log:
            dropped_lines = sum(len(x['dropped']) for x in ds_dropped_log)
            print(f"⚠️ 数据版本 {ds_name}: 自动清洗非法标签 {dropped_lines} 行，涉及 {len(ds_dropped_log)} 个文件")
            report_path = os.path.join(split_root, 'label_cleanup_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(ds_dropped_log, f, ensure_ascii=False, indent=2)
            label_clean_reports.append((ds_name, report_path, dropped_lines, len(ds_dropped_log)))

        yaml_content = {
            'path': os.path.abspath(split_root),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {i: cls_name for i, cls_name in enumerate(classes)},
        }
        y_path = os.path.join(split_root, 'data.yaml')
        os.makedirs(split_root, exist_ok=True)
        with open(y_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
        yaml_paths.append(y_path)

    with open(fp_file, 'w', encoding='utf-8') as f:
        f.write(cur_fp)
    if label_clean_reports:
        print("✅ 已生成标签清洗报告：")
        for ds_name, report_path, dropped_lines, file_count in label_clean_reports:
            print(f"   - {ds_name}: {report_path}  (清洗 {dropped_lines} 行 / {file_count} 个文件)")
    return yaml_paths


# ==================== [4a. eval_video 模式：同一段视频推理，便于对比误检] ====================
def run_eval_video(args):
    """--mode eval_video：加载权重对指定视频推理，输出检测帧数/框数，便于与训练后对比误检。"""
    import cv2
    weights = getattr(args, 'weights', None)
    source = getattr(args, 'source', '')
    conf = getattr(args, 'conf', 0.25)
    iou = getattr(args, 'iou', 0.5)
    use_track = not getattr(args, 'no_track', False)
    tracker = getattr(args, 'tracker', 'bytetrack.yaml')
    model_type = getattr(args, 'model_type', 'yolo11n')
    if not weights or not os.path.isfile(weights):
        print("❌ eval_video 需要 --weights 指向已有 .pt 文件")
        sys.exit(1)
    if not source or not os.path.isfile(source):
        print("❌ eval_video 需要 --source 指向视频文件")
        sys.exit(1)
    if model_type in ('rainforest_paper_v8s',) or is_yolov8s_ablation_model(model_type) or is_yolov8s_pt_yaml_model(model_type):
        import ultralytics_rainforest  # noqa: F401
    model = YOLO(weights)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {source}")
        sys.exit(1)
    total_frames = 0
    frames_with_det = 0
    total_detections = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        if use_track:
            res = model.track(frame, conf=conf, iou=iou, verbose=False, persist=True, tracker=tracker)[0]
        else:
            res = model.predict(frame, conf=conf, iou=iou, verbose=False)[0]
        n = len(res.boxes) if res.boxes is not None else 0
        if n > 0:
            frames_with_det += 1
            total_detections += n
    cap.release()
    print(f"\n[eval_video] 视频: {source}")
    print(f"  conf={conf} iou={iou} track={use_track} tracker={tracker}")
    print(f"  总帧数={total_frames}  有框帧数={frames_with_det}  总框数={total_detections}")
    out = {"total_frames": total_frames, "frames_with_det": frames_with_det, "total_detections": total_detections}
    log_dir = getattr(args, 'log_dir', None)
    if log_dir and os.path.isdir(log_dir):
        eval_record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mode": "eval_video",
            "weights": weights,
            "source": source,
            "conf": conf,
            "iou": iou,
            "track": use_track,
            "tracker": tracker,
            **out,
        }
        append_experiment_log(log_dir, eval_record)
        print(f"  已追加到 {log_dir}/experiment_log.json")
    return out


def _read_val_metrics_from_csv(csv_path, take_last=False):
    """
    从 Ultralytics results.csv 读 val 指标。
    take_last=True  → 读最后一行（当前 epoch）
    take_last=False → 读 mAP50(B) 最高的行（最佳 epoch）
    """
    if not os.path.isfile(csv_path):
        return None
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) < 2:
            return None
        header = [x.strip() for x in lines[0].split(',')]
        data_lines = [l for l in lines[1:] if l.strip()]
        if not data_lines:
            return None

        def _parse_row(row_str):
            parts = [x.strip() for x in row_str.split(',')]
            if len(parts) != len(header):
                return None
            return dict(zip(header, parts))

        if take_last:
            row = _parse_row(data_lines[-1])
        else:
            best_row, best_map = None, -1.0
            for raw in data_lines:
                d = _parse_row(raw)
                if d is None:
                    continue
                try:
                    v = float(d.get('metrics/mAP50(B)', '-inf'))
                    if v > best_map:
                        best_map, best_row = v, d
                except Exception:
                    pass
            row = best_row

        if row is None:
            return None
        out = {}
        for k in ('metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'):
            if k in row:
                try:
                    out[k.replace('metrics/', '').replace('(B)', '')] = float(row[k])
                except Exception:
                    pass
        return out if out else None
    except Exception:
        return None


def _git_commit_and_diff(max_diff_chars=500):
    """返回 (commit_hash, diff_stat_snippet) 或 (None, None)。"""
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root, timeout=2)
        commit = (r.stdout or "").strip() if r.returncode == 0 else None
        r2 = subprocess.run(["git", "diff", "--stat"], capture_output=True, text=True, cwd=root, timeout=2)
        diff = (r2.stdout or "")[:max_diff_chars] if r2.returncode == 0 else None
        return (commit, diff)
    except Exception:
        return (None, None)


def append_experiment_log(run_results_path, record):
    """将一条实验记录追加到 run_results_path/experiment_log.json（列表格式）。"""
    path = os.path.join(run_results_path, "experiment_log.json")
    log = []
    if os.path.isfile(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                log = json.load(f)
        except Exception:
            log = []
    if not isinstance(log, list):
        log = []
    log.append(record)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] 无法写入实验记录: {e}")


def _ablation_flags(model_type: str):
    meta = ABLATION_MODEL_METADATA.get(model_type, {})
    rst_pos = meta.get('rst_pos', 'none')
    return {
        'has_adaconv': bool(meta.get('adaconv', False)),
        'has_mscb': bool(meta.get('mscb', False)),
        'has_pam': bool(meta.get('pam', False)),
        'has_rst': rst_pos != 'none',
        'rst_pos': rst_pos,
    }


def summarize_ablation_results(results_root):
    results_root = os.path.abspath(results_root)
    log_paths = []
    for root, _, files in os.walk(results_root):
        for fn in files:
            if fn == 'experiment_log.json':
                log_paths.append(os.path.join(root, fn))
    rows = []
    for lp in log_paths:
        try:
            with open(lp, 'r', encoding='utf-8') as f:
                recs = json.load(f)
            if not isinstance(recs, list):
                continue
            for rec in recs:
                if not isinstance(rec, dict):
                    continue
                mt = rec.get('model_type')
                if mt not in (['yolov8s', 'rainforest_paper_v8s'] + list(YOLOV8S_ABLATION_TYPES)):
                    continue
                tm = rec.get('test_metrics') or {}
                vm = rec.get('val_metrics') or {}
                flags = _ablation_flags(mt) if mt in YOLOV8S_ABLATION_TYPES else {
                    'has_adaconv': mt == 'rainforest_paper_v8s',
                    'has_mscb': mt == 'rainforest_paper_v8s',
                    'has_pam': mt == 'rainforest_paper_v8s',
                    'has_rst': mt == 'rainforest_paper_v8s',
                    'rst_pos': 'p3p4p5' if mt == 'rainforest_paper_v8s' else 'none',
                }
                row = {
                    'model_type': mt,
                    'dataset_name': rec.get('dataset_name'),
                    'fold': rec.get('fold'),
                    'test_precision': tm.get('precision'),
                    'test_recall': tm.get('recall'),
                    'test_mAP50': tm.get('mAP50'),
                    'test_mAP50_95': tm.get('mAP50-95'),
                    'val_precision': vm.get('precision'),
                    'val_recall': vm.get('recall'),
                    'val_mAP50': vm.get('mAP50'),
                    'val_mAP50_95': vm.get('mAP50-95'),
                    'params_snapshot': rec.get('params_snapshot'),
                    'loss_weights': rec.get('loss_weights'),
                    'finetune_epochs': rec.get('finetune_epochs'),
                    'pretrain_report': rec.get('pretrain_transfer_report'),
                    'contains_adaconv': flags['has_adaconv'],
                    'contains_mscb': flags['has_mscb'],
                    'contains_pam': flags['has_pam'],
                    'contains_rst': flags['has_rst'],
                    'rst_position': flags['rst_pos'],
                }
                rows.append(row)
        except Exception:
            continue
    if not rows:
        print(f"[AblationSummary] 未找到结果: {results_root}")
        return

    baseline_map = {}
    for r in rows:
        if r['model_type'] == 'yolov8s':
            baseline_map[(r['dataset_name'], r['fold'])] = r
    for r in rows:
        b = baseline_map.get((r['dataset_name'], r['fold']))
        for metric_key, out_key in (('test_recall', 'delta_recall_vs_yolov8s'),
                                    ('test_mAP50', 'delta_mAP50_vs_yolov8s'),
                                    ('test_mAP50_95', 'delta_mAP50_95_vs_yolov8s')):
            if b and r.get(metric_key) is not None and b.get(metric_key) is not None:
                r[out_key] = float(r[metric_key]) - float(b[metric_key])
            else:
                r[out_key] = None

    rows.sort(key=lambda x: (str(x.get('dataset_name')), -(x.get('test_mAP50_95') or -1.0)))
    out_json = os.path.join(results_root, 'ablation_summary.json')
    out_csv = os.path.join(results_root, 'ablation_summary.csv')
    out_md = os.path.join(results_root, 'ablation_summary.md')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    headers = [
        'model_type', 'dataset_name', 'fold', 'test_precision', 'test_recall', 'test_mAP50', 'test_mAP50_95',
        'delta_recall_vs_yolov8s', 'delta_mAP50_vs_yolov8s', 'delta_mAP50_95_vs_yolov8s',
        'contains_adaconv', 'contains_mscb', 'contains_pam', 'contains_rst', 'rst_position'
    ]
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write(','.join(headers) + '\n')
        for r in rows:
            vals = []
            for h in headers:
                v = r.get(h)
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v) if v is not None else '')
            f.write(','.join(vals) + '\n')

    def _avg(model_type, metric):
        vals = [float(r[metric]) for r in rows if r['model_type'] == model_type and r.get(metric) is not None]
        return (sum(vals) / len(vals)) if vals else None

    avg = {m: {'recall': _avg(m, 'test_recall'), 'map50': _avg(m, 'test_mAP50'), 'map5095': _avg(m, 'test_mAP50_95')}
           for m in set(r['model_type'] for r in rows)}
    def _diff(a, b, k):
        if avg.get(a, {}).get(k) is None or avg.get(b, {}).get(k) is None:
            return None
        return avg[a][k] - avg[b][k]

    analysis_lines = ["## 模块影响分析"]
    if (_diff('yolov8s_ablate_no_rst', 'yolov8s_ablate_full', 'map5095') or -1) > 0.01:
        analysis_lines.append("- RST 是主要负贡献模块，尤其需要继续检查 P3/P4/P5 分支。")
    if (_diff('yolov8s_ablate_rst_p5_only', 'yolov8s_ablate_no_rst', 'map5095') or 999) > -0.01 and (_diff('yolov8s_ablate_full', 'yolov8s_ablate_no_rst', 'map5095') or 1) < -0.01:
        analysis_lines.append("- RST 可以尝试只保留在 P5，高层语义分支可用，但不宜放在 P3 小目标分支。")
    if (_diff('yolov8s_ablate_adaconv_only', 'yolov8s', 'map5095') or 1) < -0.01:
        analysis_lines.append("- 全量 C2f_AdaConv 替换削弱预训练迁移，建议只在高层替换。")
    if (_diff('yolov8s_ablate_mscb_pam', 'yolov8s', 'map5095') or -1) >= -0.005:
        analysis_lines.append("- MSCB + PAM 是较稳定的轻量改进组合，可作为最终改进模型候选。")
    if (_diff('yolov8s_ablate_pam_only', 'yolov8s', 'map5095') or -1) > 0:
        analysis_lines.append("- PAM 单独引入在当前数据上有正向增益。")
    if (_diff('yolov8s_ablate_mscb_only', 'yolov8s', 'map5095') or -1) > 0:
        analysis_lines.append("- MSCB 单独引入在当前数据上有正向增益。")
    all_worse = True
    for m in YOLOV8S_ABLATION_TYPES:
        d = _diff(m, 'yolov8s', 'map5095')
        if d is not None and d >= 0:
            all_worse = False
            break
    if all_worse:
        analysis_lines.append("- 当前数据集上 YOLOv8s baseline 已较强，完整结构改进不适配，应以 baseline 或轻量结构作为最终模型。")

    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# Ablation Summary\n\n")
        f.write("| model_type | dataset_name | fold | test_precision | test_recall | test_mAP50 | test_mAP50_95 | delta_recall_vs_yolov8s | delta_mAP50_vs_yolov8s | delta_mAP50_95_vs_yolov8s | contains_adaconv | contains_mscb | contains_pam | contains_rst | rst_position |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|\n")
        for r in rows:
            p = '' if r['test_precision'] is None else f"{float(r['test_precision']):.4f}"
            rc = '' if r['test_recall'] is None else f"{float(r['test_recall']):.4f}"
            m50 = '' if r['test_mAP50'] is None else f"{float(r['test_mAP50']):.4f}"
            m95 = '' if r['test_mAP50_95'] is None else f"{float(r['test_mAP50_95']):.4f}"
            dr = '' if r['delta_recall_vs_yolov8s'] is None else f"{float(r['delta_recall_vs_yolov8s']):+.4f}"
            dm50 = '' if r['delta_mAP50_vs_yolov8s'] is None else f"{float(r['delta_mAP50_vs_yolov8s']):+.4f}"
            dm95 = '' if r['delta_mAP50_95_vs_yolov8s'] is None else f"{float(r['delta_mAP50_95_vs_yolov8s']):+.4f}"
            f.write(
                f"| {r['model_type']} | {r['dataset_name']} | {r['fold']} | "
                f"{p} | {rc} | {m50} | {m95} | {dr} | {dm50} | {dm95} | "
                f"{r['contains_adaconv']} | {r['contains_mscb']} | {r['contains_pam']} | {r['contains_rst']} | {r['rst_position']} |\n"
            )
        f.write("\n")
        for ln in analysis_lines:
            f.write(ln + "\n")
    print(f"[AblationSummary] 已写入: {out_csv} / {out_json} / {out_md}")



def _extract_metrics_dict(metrics_obj):
    out = {}
    try:
        rd = getattr(metrics_obj, 'results_dict', None)
        if isinstance(rd, dict):
            for src_k, dst_k in (
                ('metrics/precision(B)', 'precision'),
                ('metrics/recall(B)', 'recall'),
                ('metrics/mAP50(B)', 'mAP50'),
                ('metrics/mAP50-95(B)', 'mAP50-95'),
            ):
                if src_k in rd:
                    try:
                        out[dst_k] = float(rd[src_k])
                    except Exception:
                        pass
    except Exception:
        pass
    if not out:
        try:
            box = getattr(metrics_obj, 'box', None)
            if box is not None:
                for src_k, dst_k in (('mp', 'precision'), ('mr', 'recall'), ('map50', 'mAP50'), ('map', 'mAP50-95')):
                    if hasattr(box, src_k):
                        out[dst_k] = float(getattr(box, src_k))
        except Exception:
            pass
    return out


def evaluate_test_split(weights_path, yaml_file, project_dir, run_name, model_type):
    if not weights_path or not os.path.isfile(weights_path):
        print(f"[TestEval] ⚠️ 权重不存在，跳过 test 评估: {weights_path}")
        return None
    try:
        if model_type in ('rainforest', 'rainforest_n', 'rainforest_v8n', 'rainforest_n_lite', 'rainforest_n_lite_plus', 'rainforest_staged') or is_yolov8s_ablation_model(model_type) or model_type in ('rainforest_paper_v8s',) or is_yolov8s_pt_yaml_model(model_type):
            import ultralytics_rainforest  # noqa: F401
        test_model = YOLO(weights_path)
        print(f"[TestEval] 开始在 test 集评估: {run_name}")
        metrics_obj = test_model.val(
            data=yaml_file,
            split='test',
            project=project_dir,
            name=f"{run_name}_test",
            plots=True,
            save_json=False,
            verbose=True,
        )
        metrics = _extract_metrics_dict(metrics_obj)
        print(f"[TestEval] test 指标: {metrics if metrics else '无法解析，已完成评估'}")
        del test_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics or None
    except Exception as e:
        print(f"[TestEval] ⚠️ test 评估失败: {e}")
        return None


def check_ema_compatibility(model):
    """
    启动期检查：EMA 参数图与模型是否一致。不一致则 raise，避免训练期静默跳过导致实验口径混乱。
    若需先跑通，应在 trainer 侧显式关闭 EMA，而非 patch 成静默跳过。
    """
    try:
        from ultralytics.utils.torch_utils import ModelEMA
        ema = ModelEMA(model)
        msd = model.state_dict()
        esd = ema.ema.state_dict()
        bad = []
        for k, v in esd.items():
            if k not in msd:
                bad.append((k, "missing"))
            elif v.shape != msd[k].shape:
                bad.append((k, f"{tuple(v.shape)} != {tuple(msd[k].shape)}"))
        if bad:
            print("[EMA] incompatible keys:")
            for x in bad[:20]:
                print("   ", x)
            if len(bad) > 20:
                print(f"   ... 共 {len(bad)} 处")
            raise RuntimeError(
                "EMA 参数图与模型不一致，请先修模型构图/注册，再训练；或在此模式下关闭 EMA。"
            )
        print("[EMA] 参数图一致，可安全使用 EMA")
    except Exception as e:
        if "EMA 参数图与模型不一致" in str(e):
            raise
        print(f"[WARN] EMA 兼容性检查跳过: {e}")


def _apply_wiou_loss_patch():
    """
    原为全局 bbox_iou 劫持（CIoU→WIoU），已取消，以保持实验口径干净。
    若需 WIoU，应在 trainer/loss 入口显式传 WIoU=True，不做全局 patch。
    """
    pass


def _unpatch_wiou_loss():
    """恢复 bbox_iou 为原始实现（定义保留；当前流程不调用，调用会导致 Phase2 损失断层）。"""
    try:
        import ultralytics.utils.metrics as um
        if not getattr(um, "_wiou_patched", False):
            return
        orig = getattr(um, "_wiou_orig_bbox_iou", None)
        if orig is not None:
            um.bbox_iou = orig
            um._wiou_patched = False
            print("[Loss] 已恢复标准 CIoU（关闭 WIoU）")
    except Exception as e:
        print(f"[WARN] WIoU unpatch 失败: {e}")


def _rainforest_to_baseline_layer_map():
    """
    yolo11s-rainforest（31层, 0-30）与 yolo11s 基线（23层, 0-22）层索引对应关系。
    新增层（随机初始化，无基线对应参数）：MSCB@7, PAM@14/18/23/28, RST@19/24/29。
    """
    return {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
        7: None,              # MSCB（新增）
        8: 7, 9: 8, 10: 9,
        11: 10, 12: 11, 13: 12,
        14: None,             # PAM #1（新增）
        15: 13, 16: 14, 17: 15,
        18: None, 19: None,   # PAM #2 / RST #1（新增）
        20: 16, 21: 17, 22: 18,
        23: None, 24: None,   # PAM #3 / RST #2（新增）
        25: 19, 26: 20, 27: 21,
        28: None, 29: None,   # PAM #4 / RST #3（新增）
        30: 22,
    }


def _rainforest_n_lite_to_baseline_layer_map():
    """
    yolo11n-rainforest-lite（27 层, 0-26）与 yolo11n 基线（23 层, 0-22）层索引对应关系。
    无 RST，仅 MSCB@7、PAM@14/18/22 为新增层。
    """
    return {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
        7: None,        # MSCB
        8: 7, 9: 8, 10: 9,
        11: 10, 12: 11, 13: 12,
        14: None,       # PAM#1
        15: 13, 16: 14, 17: 15,
        18: None,       # PAM#2
        19: 16, 20: 17, 21: 18,
        22: None,       # PAM#3
        23: 19, 24: 20, 25: 21,
        26: 22,         # Detect
    }


def _rainforest_n_lite_plus_to_baseline_layer_map():
    """
    yolo11n-rainforest-lite+（28 层, 0-27）与 yolo11n 基线（23 层, 0-22）层索引对应关系。
    在 n-lite 基础上多一层 RST@26（P5 仅 1 个 RST），Detect 在 27。
    """
    return {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
        7: None,        # MSCB
        8: 7, 9: 8, 10: 9,
        11: 10, 12: 11, 13: 12,
        14: None,       # PAM#1
        15: 13, 16: 14, 17: 15,
        18: None,       # PAM#2
        19: 16, 20: 17, 21: 18,
        22: None,       # PAM#3
        23: 19, 24: 20, 25: 21,
        26: None,       # RST P5
        27: 22,         # Detect
    }


def get_rainforest_backbone_indices(model):
    """返回 rainforest YAML 中 backbone 层的索引列表（层 0-10，含 MSCB@7）。"""
    inner = getattr(model, 'model', model)
    seq   = getattr(inner, 'model', None)
    if seq is None:
        return list(range(11))
    for idx, layer in enumerate(seq):
        if type(layer).__name__ == 'C2PSA':
            return list(range(idx + 1))
    return list(range(11))


def get_rainforest_phase1_freeze_indices(model):
    """
    Phase-1 冻结「纯继承」层（Conv / C2PSA），不冻含随机初始化的模块。
    被冻结层：0(Conv) 1(Conv) 3(Conv) 5(Conv) 8(Conv) 10(C2PSA)
    不冻结：2/4/6(C3k2_AdaConv) 7(MSCB)
    """
    PURE_NAMES = ('Conv', 'C2PSA')
    inner  = getattr(model, 'model', model)
    seq    = getattr(inner, 'model', None)
    if seq is None:
        return [0, 1, 3, 5, 8, 10]
    indices = []
    for idx, layer in enumerate(seq):
        if idx >= 11:
            break
        name = type(layer).__name__
        if name in PURE_NAMES:
            indices.append(idx)
        if name == 'C2PSA':
            break
    return indices if indices else [0, 1, 3, 5, 8, 10]


def load_rainforest_with_pretrained(yaml_path='cfg/yolo11s-rainforest.yaml', pt_path='yolo11s.pt'):
    """
    仅按「显式层映射 + suffix 完全一致 + shape 一致」加载预训练，不做同 shape 猜匹配。
    原则：宁可少加载，也不要错加载。
    """
    model = YOLO(yaml_path)
    if not os.path.isfile(pt_path):
        return model
    try:
        try:
            ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(pt_path, map_location='cpu')
        pretrained_sd = ckpt.get('model')
        if pretrained_sd is None:
            pretrained_sd = ckpt
        if hasattr(pretrained_sd, 'state_dict'):
            pretrained_sd = pretrained_sd.state_dict()
        model_sd = model.model.state_dict()
        layer_map = _rainforest_to_baseline_layer_map()

        new_sd = {}
        matched = 0
        random_init = 0
        for k, v in model_sd.items():
            if not k.startswith("model."):
                new_sd[k] = v
                continue
            rest = k[6:]
            parts = rest.split(".", 1)
            if len(parts) < 2:
                new_sd[k] = v
                random_init += 1
                continue
            try:
                rf_idx = int(parts[0])
            except ValueError:
                new_sd[k] = v
                random_init += 1
                continue
            suffix = parts[1]
            bl_idx = layer_map.get(rf_idx)
            if bl_idx is None:
                new_sd[k] = v
                random_init += 1
                continue
            cand = f"model.{bl_idx}.{suffix}"
            if cand in pretrained_sd and pretrained_sd[cand].shape == v.shape:
                new_sd[k] = pretrained_sd[cand].clone()
                matched += 1
            else:
                new_sd[k] = v
                random_init += 1
        model.model.load_state_dict(new_sd, strict=False)
        print(f"[Pretrain] matched={matched}, random_init={random_init}")
    except Exception as e:
        print(f"[WARN] 预训练加载失败，从零训练: {e}")
    return model


def load_rainforest_n_lite_with_pretrained(yaml_path='cfg/yolo11n-rainforest-lite.yaml', pt_path='yolo11n.pt'):
    """
    n-lite（27 层）从 yolo11n 基线加载预训练，仅显式层映射 + suffix 一致 + shape 一致。
    """
    model = YOLO(yaml_path)
    if not os.path.isfile(pt_path):
        return model
    try:
        try:
            ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(pt_path, map_location='cpu')
        pretrained_sd = ckpt.get('model')
        if pretrained_sd is None:
            pretrained_sd = ckpt
        if hasattr(pretrained_sd, 'state_dict'):
            pretrained_sd = pretrained_sd.state_dict()
        model_sd = model.model.state_dict()
        layer_map = _rainforest_n_lite_to_baseline_layer_map()

        new_sd = {}
        matched = 0
        random_init = 0
        for k, v in model_sd.items():
            if not k.startswith("model."):
                new_sd[k] = v
                continue
            rest = k[6:]
            parts = rest.split(".", 1)
            if len(parts) < 2:
                new_sd[k] = v
                random_init += 1
                continue
            try:
                rf_idx = int(parts[0])
            except ValueError:
                new_sd[k] = v
                random_init += 1
                continue
            suffix = parts[1]
            bl_idx = layer_map.get(rf_idx)
            if bl_idx is None:
                new_sd[k] = v
                random_init += 1
                continue
            cand = f"model.{bl_idx}.{suffix}"
            if cand in pretrained_sd and pretrained_sd[cand].shape == v.shape:
                new_sd[k] = pretrained_sd[cand].clone()
                matched += 1
            else:
                new_sd[k] = v
                random_init += 1
        model.model.load_state_dict(new_sd, strict=False)
        print(f"[Pretrain n-lite] matched={matched}, random_init={random_init}")
    except Exception as e:
        print(f"[WARN] n-lite 预训练加载失败，从零训练: {e}")
    return model


def load_rainforest_n_lite_plus_with_pretrained(yaml_path='cfg/yolo11n-rainforest-lite+.yaml', pt_path='yolo11n.pt'):
    """n-lite+（28 层，P5 仅 1 个 RST）从 yolo11n 基线加载预训练。"""
    model = YOLO(yaml_path)
    if not os.path.isfile(pt_path):
        return model
    try:
        try:
            ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(pt_path, map_location='cpu')
        pretrained_sd = ckpt.get('model')
        if pretrained_sd is None:
            pretrained_sd = ckpt
        if hasattr(pretrained_sd, 'state_dict'):
            pretrained_sd = pretrained_sd.state_dict()
        model_sd = model.model.state_dict()
        layer_map = _rainforest_n_lite_plus_to_baseline_layer_map()

        new_sd = {}
        matched = 0
        random_init = 0
        for k, v in model_sd.items():
            if not k.startswith("model."):
                new_sd[k] = v
                continue
            rest = k[6:]
            parts = rest.split(".", 1)
            if len(parts) < 2:
                new_sd[k] = v
                random_init += 1
                continue
            try:
                rf_idx = int(parts[0])
            except ValueError:
                new_sd[k] = v
                random_init += 1
                continue
            suffix = parts[1]
            bl_idx = layer_map.get(rf_idx)
            if bl_idx is None:
                new_sd[k] = v
                random_init += 1
                continue
            cand = f"model.{bl_idx}.{suffix}"
            if cand in pretrained_sd and pretrained_sd[cand].shape == v.shape:
                new_sd[k] = pretrained_sd[cand].clone()
                matched += 1
            else:
                new_sd[k] = v
                random_init += 1
        model.model.load_state_dict(new_sd, strict=False)
        print(f"[Pretrain n-lite+] matched={matched}, random_init={random_init}")
    except Exception as e:
        print(f"[WARN] n-lite+ 预训练加载失败，从零训练: {e}")
    return model


def _rainforest_n_to_baseline_layer_map():
    """
    yolo11n-rainforest（32 层, 0-31）与 yolo11n 基线层索引对应关系。
    完整版 4 项改进（C3k2_AdaConv + MSCB + 4×PAM + 3×RST），且保留 SPPF。
    新增层（随机初始化）：MSCB@7, PAM@15/19/24/29, RST@20/25/30。
    比 n-lite 多 SPPF@10 + 3×RST + 1×PAM = 5 层（32 vs 27）。
    """
    return {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
        7: None,                    # MSCB（新增）
        8: 7, 9: 8,                 # Conv, C3k2_AdaConv
        10: 9, 11: 10,             # SPPF→SPPF, C2PSA→C2PSA（精确匹配！）
        12: 11, 13: 12, 14: 13,    # Upsample, Concat, C3k2
        15: None,                   # PAM#1（新增）
        16: 14, 17: 15, 18: 16,    # Upsample, Concat, C3k2
        19: None, 20: None,         # PAM#2, RST#1（新增）
        21: 17, 22: 18, 23: 19,    # Conv, Concat, C3k2
        24: None, 25: None,         # PAM#3, RST#2（新增）
        26: 20, 27: 21, 28: 22,    # Conv, Concat, C3k2
        29: None, 30: None,         # PAM#4, RST#3（新增）
        31: 23,                     # Detect
    }


def load_rainforest_n_with_pretrained(yaml_path='cfg/yolo11n-rainforest.yaml', pt_path='yolo11n.pt'):
    """
    yolo11n-rainforest 完整版（32 层）从 yolo11n 基线加载预训练。
    比 n-lite 多 SPPF 层，预训练匹配度更高（SPPF + C2PSA 均精确匹配）。
    """
    model = YOLO(yaml_path)
    if not os.path.isfile(pt_path):
        return model
    try:
        try:
            ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(pt_path, map_location='cpu')
        pretrained_sd = ckpt.get('model')
        if pretrained_sd is None:
            pretrained_sd = ckpt
        if hasattr(pretrained_sd, 'state_dict'):
            pretrained_sd = pretrained_sd.state_dict()
        model_sd = model.model.state_dict()
        layer_map = _rainforest_n_to_baseline_layer_map()

        new_sd = {}
        matched = 0
        random_init = 0
        for k, v in model_sd.items():
            if not k.startswith("model."):
                new_sd[k] = v
                continue
            rest = k[6:]
            parts = rest.split(".", 1)
            if len(parts) < 2:
                new_sd[k] = v
                random_init += 1
                continue
            try:
                rf_idx = int(parts[0])
            except ValueError:
                new_sd[k] = v
                random_init += 1
                continue
            suffix = parts[1]
            bl_idx = layer_map.get(rf_idx)
            if bl_idx is None:
                new_sd[k] = v
                random_init += 1
                continue
            cand = f"model.{bl_idx}.{suffix}"
            if cand in pretrained_sd and pretrained_sd[cand].shape == v.shape:
                new_sd[k] = pretrained_sd[cand].clone()
                matched += 1
            else:
                new_sd[k] = v
                random_init += 1
        model.model.load_state_dict(new_sd, strict=False)
        print(f"[Pretrain n-full] matched={matched}, random_init={random_init}")
    except Exception as e:
        print(f"[WARN] n-full 预训练加载失败，从零训练: {e}")
    return model


def _rainforest_v8n_to_baseline_layer_map():
    """
    yolov8n-rainforest（31 层, 0-30）与 yolov8n 基线（23 层, 0-22）层索引对应关系。
    论文最忠实版本：C2f_AdaConv + MSCB + 4×PAM + 3×RST，基于 YOLOv8n。
    """
    return {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
        7: None,        # MSCB（新增）
        8: 7, 9: 8, 10: 9,
        11: 10, 12: 11, 13: 12,
        14: None,       # PAM#1
        15: 13, 16: 14, 17: 15,
        18: None, 19: None,   # PAM#2, RST#1
        20: 16, 21: 17, 22: 18,
        23: None, 24: None,   # PAM#3, RST#2
        25: 19, 26: 20, 27: 21,
        28: None, 29: None,   # PAM#4, RST#3
        30: 22,         # Detect
    }


def _rainforest_v8s_paper_to_baseline_layer_map():
    """
    YOLOv8s paper faithful Rainforest 结构与 yolov8s 基线层索引对应关系。
    拓扑与 v8n 版一致：新增层（随机初始化）为 MSCB / PAM / RST，
    其余层按 stage 语义映射到同族 yolov8s 基线。
    由于 YOLOv8n / v8s 拓扑一致、仅宽深系数不同，这里沿用同一层级映射。
    """
    return _rainforest_v8n_to_baseline_layer_map()


def _resolve_local_path(path_str: str) -> str:
    """优先使用传入路径；若为相对路径则相对当前脚本目录解析。"""
    if os.path.isabs(path_str):
        return path_str
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path_str)


def load_yolov8s_ablation_with_pretrained(
    yaml_path: str,
    pt_path: str = 'yolov8s.pt',
    model_name: str = '',
):
    """
    通用 YOLOv8s 消融预训练迁移：
    1) 完全键名 + shape 一致优先
    2) 不一致时尝试同 suffix + shape 的保守映射
    3) 自定义模块无法匹配参数保持随机初始化
    """
    yaml_path_abs = _resolve_local_path(yaml_path)
    pt_path_abs = _resolve_local_path(pt_path)
    model = YOLO(yaml_path_abs)

    report = {
        'model_type': model_name or 'unknown_ablation',
        'yaml_path': yaml_path_abs,
        'pt_path': pt_path_abs,
        'matched': 0,
        'random_init': 0,
        'matched_ratio': 0.0,
        'random_init_keys': [],
        'skipped_due_to_shape': [],
    }

    if not os.path.isfile(pt_path_abs):
        print(f"[Pretrain ablation:{model_name}] WARNING: 预训练权重不存在 -> 从零训练: {pt_path_abs}")
        model._pretrain_transfer_report = report  # type: ignore[attr-defined]
        return model

    try:
        try:
            ckpt = torch.load(pt_path_abs, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(pt_path_abs, map_location='cpu')
        pretrained_sd = ckpt.get('model') if isinstance(ckpt, dict) else ckpt
        if pretrained_sd is None:
            pretrained_sd = ckpt
        if hasattr(pretrained_sd, 'state_dict'):
            pretrained_sd = pretrained_sd.state_dict()

        model_sd = model.model.state_dict()
        new_sd = {}
        used_pretrained_keys = set()
        skipped_due_to_shape = []

        # suffix 候选池：按 (suffix, shape) 建立映射
        suffix_pool = {}
        for pk, pv in pretrained_sd.items():
            if not pk.startswith('model.'):
                continue
            parts = pk.split('.', 2)
            if len(parts) < 3:
                continue
            suffix = parts[2]
            key = (suffix, tuple(pv.shape))
            suffix_pool.setdefault(key, []).append(pk)

        matched = 0
        random_init = 0
        random_init_keys = []

        for mk, mv in model_sd.items():
            loaded = False
            # A) 完全键名匹配
            pv = pretrained_sd.get(mk, None)
            if pv is not None:
                if tuple(pv.shape) == tuple(mv.shape):
                    new_sd[mk] = pv.clone()
                    used_pretrained_keys.add(mk)
                    matched += 1
                    loaded = True
                else:
                    skipped_due_to_shape.append({'model_key': mk, 'pretrain_key': mk, 'model_shape': list(mv.shape), 'pretrain_shape': list(pv.shape)})

            # B) 同 suffix + shape 匹配（兼容 layer index 变化）
            if not loaded and mk.startswith('model.'):
                parts = mk.split('.', 2)
                if len(parts) >= 3:
                    suffix = parts[2]
                    key = (suffix, tuple(mv.shape))
                    cands = suffix_pool.get(key, [])
                    for cand in cands:
                        if cand in used_pretrained_keys:
                            continue
                        cand_v = pretrained_sd[cand]
                        if tuple(cand_v.shape) == tuple(mv.shape):
                            new_sd[mk] = cand_v.clone()
                            used_pretrained_keys.add(cand)
                            matched += 1
                            loaded = True
                            break

            if not loaded:
                new_sd[mk] = mv
                random_init += 1
                if len(random_init_keys) < 50:
                    random_init_keys.append(mk)

        model.model.load_state_dict(new_sd, strict=False)
        total = matched + random_init
        ratio = (100.0 * matched / total) if total > 0 else 0.0
        report.update({
            'matched': matched,
            'random_init': random_init,
            'matched_ratio': ratio,
            'random_init_keys': random_init_keys[:50],
            'skipped_due_to_shape': skipped_due_to_shape[:50],
        })
        print(f"[Pretrain ablation:{model_name}] matched={matched}, random_init={random_init}, matched_ratio={ratio:.2f}%")
    except Exception as e:
        print(f"[Pretrain ablation:{model_name}] WARNING: 迁移失败，从零训练: {e}")
    model._pretrain_transfer_report = report  # type: ignore[attr-defined]
    return model


def load_yolov8s_p2_bifpn_lite_with_pretrained(
    yaml_path: str = 'cfg/yolov8s-p2-bifpn-lite.yaml',
    pt_path: str = 'yolov8s.pt',
    model_name: str = 'yolov8s_p2_bifpn_lite',
):
    """
    YOLOv8s-P2-BiFPN-lite：yolov8s.pt 迁移 + 少量 C2f 层号手动映射。
    BiFPNLite / P2 head / 4-scale Detect 保持随机初始化。
    """
    yaml_path_abs = _resolve_local_path(yaml_path)
    pt_path_abs = _resolve_local_path(pt_path)
    model = YOLO(yaml_path_abs)

    report = {
        'model_type': model_name,
        'yaml_path': yaml_path_abs,
        'pt_path': pt_path_abs,
        'matched': 0,
        'random_init': 0,
        'matched_ratio': 0.0,
        'random_init_keys': [],
        'skipped_due_to_shape': [],
    }

    if not os.path.isfile(pt_path_abs):
        print(f"[Pretrain {model_name}] WARNING: 预训练权重不存在 -> 从零训练: {pt_path_abs}")
        model._pretrain_transfer_report = report  # type: ignore[attr-defined]
        return model

    # 官方 v8s head C2f: 12(td P4), 15(td P3), 18(bu P4), 21(bu P5)
    # P2-BiFPN-lite: 16(td P3), 28(bu P4), 32(bu P5)
    manual_layer_map = {16: 15, 28: 18, 32: 21}

    try:
        try:
            ckpt = torch.load(pt_path_abs, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(pt_path_abs, map_location='cpu')
        pretrained_sd = ckpt.get('model') if isinstance(ckpt, dict) else ckpt
        if pretrained_sd is None:
            pretrained_sd = ckpt
        if hasattr(pretrained_sd, 'state_dict'):
            pretrained_sd = pretrained_sd.state_dict()

        model_sd = model.model.state_dict()
        new_sd = {}
        matched = 0
        random_init = 0
        random_init_keys = []
        skipped_due_to_shape = []

        for k, v in model_sd.items():
            loaded = False
            pv = pretrained_sd.get(k)
            if pv is not None and tuple(pv.shape) == tuple(v.shape):
                new_sd[k] = pv.clone()
                matched += 1
                loaded = True
            elif k.startswith('model.'):
                rest = k[6:]
                parts = rest.split('.', 1)
                if len(parts) == 2:
                    try:
                        cur_idx = int(parts[0])
                        suffix = parts[1]
                        src_idx = manual_layer_map.get(cur_idx)
                        if src_idx is not None:
                            cand = f'model.{src_idx}.{suffix}'
                            pv2 = pretrained_sd.get(cand)
                            if pv2 is not None and tuple(pv2.shape) == tuple(v.shape):
                                new_sd[k] = pv2.clone()
                                matched += 1
                                loaded = True
                            elif pv2 is not None:
                                skipped_due_to_shape.append({
                                    'model_key': k,
                                    'pretrain_key': cand,
                                    'model_shape': list(v.shape),
                                    'pretrain_shape': list(pv2.shape),
                                })
                    except ValueError:
                        pass
            if not loaded:
                new_sd[k] = v
                random_init += 1
                if len(random_init_keys) < 80:
                    random_init_keys.append(k)

        model.model.load_state_dict(new_sd, strict=False)
        total = matched + random_init
        ratio = (100.0 * matched / total) if total > 0 else 0.0
        report.update({
            'matched': matched,
            'random_init': random_init,
            'matched_ratio': ratio,
            'random_init_keys': random_init_keys,
            'skipped_due_to_shape': skipped_due_to_shape[:80],
        })
        print(
            f"[Pretrain {model_name}] matched={matched}, random_init={random_init}, "
            f"matched_ratio={ratio:.2f}%"
        )
    except Exception as e:
        print(f"[Pretrain {model_name}] WARNING: 迁移失败，从零训练: {e}")

    model._pretrain_transfer_report = report  # type: ignore[attr-defined]
    return model


def load_rainforest_v8s_paper_with_pretrained(yaml_path='cfg/yolov8s-rainforest-paper.yaml', pt_path='yolov8s.pt'):
    """
    Rainforest-YOLO(v8s paper faithful) 从 yolov8s 基线加载预训练。
    仅按“显式层映射 + suffix 完全一致 + shape 一致”加载；新增模块随机初始化。
    """
    yaml_path_abs = _resolve_local_path(yaml_path)
    pt_path_abs = _resolve_local_path(pt_path)
    model = YOLO(yaml_path_abs)
    if not os.path.isfile(pt_path_abs):
        print(f"[INFO] 预训练权重 {pt_path_abs} 不存在，Ultralytics 会自动下载")
        return model
    try:
        try:
            ckpt = torch.load(pt_path_abs, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(pt_path_abs, map_location='cpu')
        pretrained_sd = ckpt.get('model')
        if pretrained_sd is None:
            pretrained_sd = ckpt
        if hasattr(pretrained_sd, 'state_dict'):
            pretrained_sd = pretrained_sd.state_dict()
        model_sd = model.model.state_dict()
        layer_map = _rainforest_v8s_paper_to_baseline_layer_map()

        new_sd = {}
        matched = 0
        random_init = 0
        for k, v in model_sd.items():
            if not k.startswith("model."):
                new_sd[k] = v
                continue
            rest = k[6:]
            parts = rest.split(".", 1)
            if len(parts) < 2:
                new_sd[k] = v
                random_init += 1
                continue
            try:
                rf_idx = int(parts[0])
            except ValueError:
                new_sd[k] = v
                random_init += 1
                continue
            suffix = parts[1]
            bl_idx = layer_map.get(rf_idx)
            if bl_idx is None:
                new_sd[k] = v
                random_init += 1
                continue
            cand = f"model.{bl_idx}.{suffix}"
            if cand in pretrained_sd and pretrained_sd[cand].shape == v.shape:
                new_sd[k] = pretrained_sd[cand].clone()
                matched += 1
            else:
                new_sd[k] = v
                random_init += 1
        model.model.load_state_dict(new_sd, strict=False)
        print(f"[Pretrain v8s-paper] matched={matched}, random_init={random_init}")
    except Exception as e:
        print(f"[WARN] v8s-paper 预训练加载失败，从零训练: {e}")
    return model


def load_rainforest_v8n_with_pretrained(yaml_path='cfg/yolov8n-rainforest.yaml', pt_path='yolov8n.pt'):
    """
    YOLOv8n-rainforest（31 层）从 yolov8n 基线加载预训练。
    C2f_AdaConv 的 cv1/cv2 与 C2f 精确匹配，gate MLP 随机初始化。
    """
    model = YOLO(yaml_path)
    if not os.path.isfile(pt_path):
        print(f"[INFO] 预训练权重 {pt_path} 不存在，Ultralytics 会自动下载")
        return model
    try:
        try:
            ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(pt_path, map_location='cpu')
        pretrained_sd = ckpt.get('model')
        if pretrained_sd is None:
            pretrained_sd = ckpt
        if hasattr(pretrained_sd, 'state_dict'):
            pretrained_sd = pretrained_sd.state_dict()
        model_sd = model.model.state_dict()
        layer_map = _rainforest_v8n_to_baseline_layer_map()

        new_sd = {}
        matched = 0
        random_init = 0
        for k, v in model_sd.items():
            if not k.startswith("model."):
                new_sd[k] = v
                continue
            rest = k[6:]
            parts = rest.split(".", 1)
            if len(parts) < 2:
                new_sd[k] = v
                random_init += 1
                continue
            try:
                rf_idx = int(parts[0])
            except ValueError:
                new_sd[k] = v
                random_init += 1
                continue
            suffix = parts[1]
            bl_idx = layer_map.get(rf_idx)
            if bl_idx is None:
                new_sd[k] = v
                random_init += 1
                continue
            cand = f"model.{bl_idx}.{suffix}"
            if cand in pretrained_sd and pretrained_sd[cand].shape == v.shape:
                new_sd[k] = pretrained_sd[cand].clone()
                matched += 1
            else:
                new_sd[k] = v
                random_init += 1
        model.model.load_state_dict(new_sd, strict=False)
        print(f"[Pretrain v8n-rf] matched={matched}, random_init={random_init}")
    except Exception as e:
        print(f"[WARN] v8n-rainforest 预训练加载失败，从零训练: {e}")
    return model


# ==================== [3b. Rainforest 分阶段渐进解冻训练] ====================
# 消融表明四模块同时随机初始化互相干扰；渐进解冻让每组模块在稳定特征上依次收敛。
# Phase 1: 冻结 MSCB/PAM/RST，训 Backbone+AdaConv+Head
# Phase 2: 解冻 MSCB+PAM，仍冻结 RST
# Phase 3: 全部解冻
# 用法: --model_type rainforest_staged [--single_fold] [--phases 1 2 3] [--resume_phase N] [--dry_run]

CUSTOM_MODULE_TYPES = {"C3k2_AdaConv", "MSCB", "PAM", "ResPAM", "RST", "ST"}


def _get_sequential_staged(model):
    inner = getattr(model, 'model', model)
    seq = getattr(inner, 'model', None)
    if seq is None:
        raise RuntimeError("无法访问 model.model.model (nn.Sequential)")
    return seq


def discover_module_indices(model):
    seq = _get_sequential_staged(model)
    result = {}
    for idx, layer in enumerate(seq):
        cls_name = type(layer).__name__
        if cls_name in CUSTOM_MODULE_TYPES:
            result.setdefault(cls_name, []).append(idx)
    return result


def get_freeze_indices_staged(module_map, freeze_type_names):
    indices = []
    for name in freeze_type_names:
        if name in module_map:
            indices.extend(module_map[name])
    return sorted(set(indices))


def print_model_architecture_staged(model, module_map, freeze_indices=None):
    seq = _get_sequential_staged(model)
    freeze_set = set(freeze_indices or [])
    total, frozen_total = 0, 0
    for idx, layer in enumerate(seq):
        cls_name = type(layer).__name__
        n = sum(p.numel() for p in layer.parameters())
        total += n
        if idx in freeze_set:
            frozen_total += n
    print(f"  总参数: {total:,}  冻结: {frozen_total:,}  可训练: {total - frozen_total:,}\n")


def verify_module_init_staged(model, module_map):
    seq = _get_sequential_staged(model)
    for idx in module_map.get("MSCB", []):
        layer = seq[idx]
        if hasattr(layer, 'proj') and hasattr(layer.proj, 'bias') and layer.proj.bias is not None:
            b = layer.proj.bias.data.mean().item()
            print(f"     MSCB[{idx}] proj.bias={b:.4f} -> sigmoid={torch.sigmoid(torch.tensor(b)).item():.4f}")
    for idx in module_map.get("PAM", []):
        print(f"     PAM[{idx}] CAM/SAM 无 bias -> 恒等")
    for idx in module_map.get("RST", []):
        print(f"     RST[{idx}] 大残差 -> 初始输出≈2x, BN 吸收")
    print("  ✅ 冻结模块初始化验证完成\n")


def capture_weight_fingerprint(model, target_indices=None):
    inner = getattr(model, 'model', model)
    fp = {}
    for name, param in inner.named_parameters():
        if target_indices is not None:
            parts = name.split(".")
            if len(parts) >= 2 and parts[0] == "model":
                try:
                    idx = int(parts[1])
                except ValueError:
                    continue
                if idx not in target_indices:
                    continue
        data = param.detach().float().cpu()
        fp[name] = {"hash": hashlib.md5(data.numpy().tobytes()).hexdigest()[:16], "mean": data.mean().item(), "std": data.std().item()}
    return fp


def verify_weight_continuity_staged(fp_prev, fp_curr, prev_freeze_indices, tag=""):
    unchanged_ok, unchanged_fail = 0, 0
    for name in fp_prev:
        if name not in fp_curr:
            continue
        parts = name.split(".")
        if len(parts) < 2 or parts[0] != "model":
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        same_hash = (fp_prev[name]["hash"] == fp_curr[name]["hash"])
        if idx in prev_freeze_indices:
            if same_hash:
                unchanged_ok += 1
            else:
                unchanged_fail += 1
    print(f"  [{tag}] 冻结层: {unchanged_ok} 未变, {unchanged_fail} 异常")
    return unchanged_fail == 0


def run_staged_training(args, yaml_files=None):
    import ultralytics_rainforest  # noqa: F401
    phases_to_run = sorted(set(args.phases)) if getattr(args, 'phases', None) else [1, 2, 3]
    if getattr(args, 'resume_phase', None):
        phases_to_run = [p for p in phases_to_run if p >= args.resume_phase]

    _base = os.path.abspath(args.output_dir) if args.single_fold else os.path.abspath(results_path_rainforest)
    run_results_path = os.path.join(_base, "staged_results")
    os.makedirs(run_results_path, exist_ok=True)
    data_yamls = yaml_files if yaml_files is not None else prepare_kfold_data()
    all_bad = []
    for yf in data_yamls:
        for sub in ('train', 'val', 'test'):
            lbl_dir = os.path.join(os.path.dirname(yf), sub, 'labels')
            all_bad.extend(check_yolo_labels(lbl_dir))
    if all_bad:
        print(f"❌ 标签异常 {len(all_bad)} 条")
        sys.exit(1)
    _apply_wiou_loss_patch()

    for fold_i, yaml_file in enumerate(data_yamls):
        fold_num = fold_i + 1
        dataset_name = _extract_dataset_name_from_yaml(yaml_file)
        fold_dir = os.path.join(run_results_path, dataset_name)
        for root, _, files in os.walk(os.path.dirname(yaml_file)):
            for f in files:
                if f.endswith('.cache'):
                    os.remove(os.path.join(root, f))
        prev_best_pt = None
        prev_fingerprint = None
        prev_freeze_idx_set = set()

        for phase in phases_to_run:
            cfg = PHASE_CONFIGS[phase]
            phase_dir_name = f"phase{phase}"
            print(f"\n{'─'*55}\n  Phase {phase}/3 — Fold {fold_num}  epochs={cfg['epochs']}  lr0={cfg['lr0']}  freeze={cfg['freeze_module_types'] or '无'}\n{'─'*55}")

            if phase == 1 and prev_best_pt is None:
                model = load_rainforest_with_pretrained('cfg/yolo11s-rainforest.yaml', 'yolo11s.pt')
            else:
                pt_path = prev_best_pt
                if not pt_path or not os.path.isfile(pt_path):
                    print("❌ 找不到前一阶段 best.pt"); sys.exit(1)
                model = YOLO(pt_path)

            check_ema_compatibility(model.model)
            module_map = discover_module_indices(model)
            if not module_map:
                print("❌ 未检测到自定义模块"); sys.exit(1)
            freeze_indices = get_freeze_indices_staged(module_map, cfg["freeze_module_types"])
            print_model_architecture_staged(model, module_map, freeze_indices)
            if phase == 1 and prev_best_pt is None:
                verify_module_init_staged(model, module_map)
            if phase > 1 and prev_fingerprint is not None:
                curr_fp = capture_weight_fingerprint(model)
                verify_weight_continuity_staged(prev_fingerprint, curr_fp, prev_freeze_idx_set, tag=f"P{phase-1}->P{phase}")

            if getattr(args, 'dry_run', False):
                prev_fingerprint = capture_weight_fingerprint(model)
                prev_freeze_idx_set = set(freeze_indices)
                del model
                gc.collect()
                continue

            train_kw = {
                **STAGED_SHARED_PARAMS,
                'data': yaml_file,
                'project': fold_dir,
                'name': phase_dir_name,
                'plots': True,
                'seed': 42 + fold_num + phase * 100,
                'deterministic': False,
                'pretrained': False,
                'epochs': cfg['epochs'],
                'lr0': cfg['lr0'], 'lrf': cfg['lrf'], 'warmup_epochs': cfg['warmup_epochs'],
                'optimizer': cfg['optimizer'], 'weight_decay': cfg['weight_decay'], 'cos_lr': cfg['cos_lr'],
                'mosaic': cfg['mosaic'], 'close_mosaic': cfg['close_mosaic'],
                'mixup': cfg['mixup'], 'copy_paste': cfg['copy_paste'],
                'box': cfg['box'], 'cls': cfg['cls'], 'dfl': cfg['dfl'],
            }
            if freeze_indices:
                train_kw['freeze'] = freeze_indices

            def _grad_clip(trainer):
                try:
                    if trainer.model is not None:
                        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=10.0)
                except Exception:
                    pass
            add_cb = getattr(model, 'add_callback', None)
            if callable(add_cb):
                add_cb('on_train_batch_end', _grad_clip)

            tb_dir_staged = os.path.join(run_results_path, f"tb_fold{fold_num}_phase{phase}")
            close_staged_profiler = install_live_profiler_ultralytics(
                model,
                logdir=tb_dir_staged,
                enable_nvtx_batch=False,
                enable_profiler=False,
            )

            print(f"  🚀 Phase {phase} 训练 ({cfg['epochs']} epochs)...")
            t0 = time.time()
            model.train(**train_kw)
            elapsed = time.time() - t0

            if close_staged_profiler is not None:
                close_staged_profiler()

            best_pt = None
            phase_save_dir = str(model.trainer.save_dir)
            cand = os.path.join(phase_save_dir, "weights", "best.pt")
            if os.path.isfile(cand):
                best_pt = cand
            else:
                cand_last = os.path.join(phase_save_dir, "weights", "last.pt")
                if os.path.isfile(cand_last):
                    best_pt = cand_last
            if best_pt:
                prev_best_pt = best_pt
                tmp = YOLO(best_pt)
                prev_fingerprint = capture_weight_fingerprint(tmp)
                del tmp
            else:
                prev_fingerprint = capture_weight_fingerprint(model)
            prev_freeze_idx_set = set(freeze_indices)

            csv_path = os.path.join(os.path.dirname(os.path.dirname(best_pt)), "results.csv") if best_pt else None
            val_metrics = _read_val_metrics_from_csv(csv_path) if csv_path else None
            test_metrics = evaluate_test_split(best_pt, yaml_file, fold_dir, f"phase{phase}", 'rainforest_staged') if best_pt else None
            record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "mode": "staged_training",
                "dataset_name": dataset_name,
                "fold": fold_num,
                "phase": phase,
                "config": {"epochs": cfg["epochs"], "lr0": cfg["lr0"], "freeze": cfg["freeze_module_types"], "box": cfg["box"], "cls": cfg["cls"], "dfl": cfg["dfl"]},
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "best_pt": best_pt,
                "elapsed_min": round(elapsed / 60, 1),
            }
            append_experiment_log(run_results_path, record)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(3)

        if prev_best_pt and os.path.isfile(prev_best_pt):
            final_dir = os.path.join(fold_dir, "weights")
            os.makedirs(final_dir, exist_ok=True)
            shutil.copy2(prev_best_pt, os.path.join(final_dir, "best.pt"))
            print(f"\n  📦 最终权重: {os.path.join(final_dir, 'best.pt')}")

    print(f"\n🎉 分阶段训练完成! 结果: {run_results_path}\n")


# ==================== [4. 主训练循环] ====================
# 若换折时报错 OSError [WinError 1455] 页面文件太小：请增大 Windows 虚拟内存（页面文件）。
# 设置方法：系统属性 → 高级 → 性能设置 → 高级 → 虚拟内存 → 自定义大小，建议至少 16GB。
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='K-Fold 训练 / 视频推理评估')
    all_train_model_choices = [
        'all',
        'yolov8s',
        'rainforest_paper_v8s',
        'yolov8s_ablate_baseline',
        'yolov8s_ablate_adaconv_only',
        'yolov8s_ablate_mscb_only',
        'yolov8s_ablate_pam_only',
        'yolov8s_ablate_rst_only',
        'yolov8s_ablate_mscb_pam',
        'yolov8s_ablate_no_rst',
        'yolov8s_ablate_rst_p5_only',
        'yolov8s_ablate_rst_p4p5',
        'yolov8s_ablate_full',
        'yolov8s_monkeyplus_p2',
        'yolov8s_monkeyplus_p2_respam',
        'yolov8s_p2_only',
        'yolov8s_p2_only_cm30',
        'yolov8s_p2_only_cm30_ft800',
        'yolov8s_p2_wide',
        'yolov8s_p2_wide_respam',
        'yolov8s_p2_respam_p2only',
        'yolov8s_p2_respam_p2only_gql_nwd',
        'yolov8s_p2_respam_decoupled_p2head',
        'yolov8s_p2_respam_hfres',
        'yolov8s_p2_respam_hfres_ft800',
        'yolov8s_p2_respam_sgf',
        'yolov8s_p2_respam_sgf_ft800',
        'yolov8s_p2_wide_respam_sgf',
        'yolov8s_p2_wide_respam_sgf_ft800',
        'yolov8s_p2_bifpn_lite',
        'yolov8s_p2_hfres',
        'yolov8s_p2_sgf',
        'yolov8s_p2_hfres_sgf',
        'yolov8s_p2_hfres_sgf_ft800',
        'yolov8s_p2_aux_p3main',
        'yolov8s_p2_auxsup_p3main',
    ]
    parser.add_argument('--mode', type=str, choices=['train', 'eval_video'], default='train',
                        help='train: 正常训练; eval_video: 仅对视频推理，对比 conf/iou/误检')
    parser.add_argument('--model_type', type=str,
                        choices=all_train_model_choices,
                        default='all',
                        help='训练模型类型（含 baseline、论文 full、以及各 ablation 变体）')
    parser.add_argument('--single_fold', action='store_true',
                        help='只训练第 1 折，结果保存到 --output_dir')
    parser.add_argument('--output_dir', type=str, default=r'D:\college\college3\monkey\yolo\result_new',
                        help='--single_fold 时的输出目录（权重在 output_dir/fold_1/weights/）')

    # ---- rainforest_staged 分阶段训练 ----
    parser.add_argument('--phases', type=int, nargs='+', default=None,
                        help='rainforest_staged: 只跑指定阶段，如 --phases 1 2')
    parser.add_argument('--resume_phase', type=int, default=None,
                        help='rainforest_staged: 从该阶段恢复，跳过更早阶段')
    parser.add_argument('--dry_run', action='store_true',
                        help='rainforest_staged: 只验证模型/冻结/初始化，不训练')

    # ---- eval_video 专用 ----
    parser.add_argument('--weights', type=str, default='',
                        help='eval_video 时必需：权重 .pt 路径')
    parser.add_argument('--source', type=str, default='test_result_fold52.mp4',
                        help='eval_video 时视频路径')
    parser.add_argument('--conf', type=float, default=0.25, help='推理置信度阈值（eval_video）')
    parser.add_argument('--iou',  type=float, default=0.5,  help='NMS IoU（eval_video）')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml',
                        help='eval_video 时 tracker（bytetrack 可跨帧补全漏检）')
    parser.add_argument('--no_track', action='store_true',
                        help='eval_video 时不用 tracker，仅 predict')
    parser.add_argument('--log_dir', type=str, default='',
                        help='eval_video 时可选，写入实验记录的目录')

    # ---- XML→YOLO 导出 ----
    parser.add_argument('--export_yolo', type=str, default='', metavar='DIR',
                        help='将当前 XML 数据导出为 YOLO 目录 DIR（含 images/ labels/），然后可运行 augment_rainforest_offline.py')

    # ---- 实时算子/数据流可视化（TensorBoard Profile）----
    parser.add_argument('--torch_profile', action='store_true',
                        help='启用 PyTorch Profiler（训练时写入 TensorBoard Profile trace）')
    parser.add_argument('--tb_logdir', type=str, default='',
                        help='TensorBoard 日志目录（建议用纯英文/ASCII 路径）')
    parser.add_argument('--profile_steps', type=int, default=200,
                        help='最多采样多少个 train batch（防止 trace 过大）')
    parser.add_argument('--nvtx_batch', action='store_true',
                        help='每个 batch 打 NVTX 标记（配合 nsys/nsight 看的更细）')

    # ---- CLI 参数覆盖（不改代码直接实验，优先级高于 CFG_*）----
    parser.add_argument('--epochs', type=int, default=None, help='覆盖训练 epoch 数（用于 1 epoch smoke）')
    parser.add_argument('--imgsz',  type=int,   default=None, help='覆盖输入边长，如 800/960')
    parser.add_argument('--batch',  type=int,   default=None, help='覆盖 batch size')
    parser.add_argument('--nbs',    type=int,   default=None, help='覆盖 nominal batch size（小 batch 稳定梯度）')
    parser.add_argument('--device', type=str,   default=None, help='覆盖训练设备，如 0 / 1 / cpu')
    parser.add_argument('--box',    type=float, default=None, help='覆盖 box 损失权重')
    parser.add_argument('--dfl',    type=float, default=None, help='覆盖 dfl 损失权重')
    parser.add_argument('--cls',    type=float, default=None, help='覆盖 cls 损失权重')
    parser.add_argument('--mosaic', type=float, default=None, help='覆盖 mosaic 概率')
    parser.add_argument('--mixup',  type=float, default=None, help='覆盖 mixup 概率')
    parser.add_argument('--close_mosaic', type=int, default=None, help='覆盖最后 N epoch 关闭 mosaic')
    parser.add_argument('--finetune_epochs', type=int, default=None,
                        help='主训练后追加 N epoch 精修（mosaic=0，低 lr），None=沿用 CFG_RAINFOREST.finetune 配置')
    parser.add_argument('--rerun_completed', action='store_true',
                        help='all 模式下强制重跑已完成模型；默认只补跑未完成模型')
    parser.add_argument('--ablation_suite', action='store_true',
                        help='按固定顺序串行运行全套消融（每个模型独立子进程）')
    parser.add_argument('--ablation_models', type=str, default='',
                        help='指定消融列表，逗号分隔，例如 yolov8s,yolov8s_ablate_no_rst')
    parser.add_argument('--summarize_ablation', action='store_true',
                        help='仅汇总已有结果，输出 ablation_summary.{csv,json,md}')
    parser.add_argument('--results_root', type=str, default='',
                        help='汇总结果根目录（--summarize_ablation 时使用）')

    args = parser.parse_args()

    if args.summarize_ablation:
        root = args.results_root or args.output_dir
        summarize_ablation_results(root)
        sys.exit(0)


    def _model_run_complete(out_dir):
        """判断某模型是否已经完整跑完 ordered 数据源。"""
        log_path = os.path.join(out_dir, 'experiment_log.json')
        if not os.path.isfile(log_path):
            return False
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                log = json.load(f)
            if not isinstance(log, list):
                return False
            done = set()
            for rec in log:
                if not isinstance(rec, dict):
                    continue
                ds = rec.get('dataset_name')
                tm = rec.get('test_metrics')
                vm = rec.get('val_metrics')
                if ds in ('ordered',) and (tm is not None or vm is not None):
                    done.add(ds)
            return {'ordered'}.issubset(done)
        except Exception:
            return False

    def _infer_incomplete_models(all_models, base_output_dir):
        incomplete = []
        completed = []
        for mt in all_models:
            out_dir = os.path.join(os.path.abspath(base_output_dir), mt)
            if _model_run_complete(out_dir):
                completed.append(mt)
            else:
                incomplete.append(mt)
        return incomplete, completed

    def _run_models_as_subprocess(model_list, base_output_dir):
        failed = []
        passthrough = []
        for flag in ('--single_fold', '--dry_run', '--torch_profile', '--nvtx_batch', '--no_track'):
            attr = flag.lstrip('-').replace('-', '_')
            if getattr(args, attr, False):
                passthrough.append(flag)
        value_flags = {
            '--mode': 'train',
            '--phases': args.phases,
            '--resume_phase': args.resume_phase,
            '--weights': args.weights,
            '--source': args.source,
            '--conf': args.conf,
            '--iou': args.iou,
            '--tracker': args.tracker,
            '--log_dir': args.log_dir,
            '--export_yolo': args.export_yolo,
            '--tb_logdir': args.tb_logdir,
            '--profile_steps': args.profile_steps,
            '--imgsz': args.imgsz,
            '--batch': args.batch,
            '--nbs': args.nbs,
            '--device': args.device,
            '--box': args.box,
            '--dfl': args.dfl,
            '--cls': args.cls,
            '--mosaic': args.mosaic,
            '--mixup': args.mixup,
            '--close_mosaic': args.close_mosaic,
            '--finetune_epochs': args.finetune_epochs,
        }
        for mt in model_list:
            out_dir = os.path.join(os.path.abspath(base_output_dir), mt)
            cmd = [sys.executable, os.path.abspath(__file__), '--mode', 'train', '--model_type', mt, '--output_dir', out_dir]
            if args.single_fold:
                cmd.append('--single_fold')
            for flag, val in value_flags.items():
                if val is None or val == '' or flag == '--mode':
                    continue
                if flag == '--phases' and isinstance(val, list):
                    cmd.append(flag)
                    cmd.extend(str(x) for x in val)
                else:
                    cmd.extend([flag, str(val)])
            for flag in passthrough:
                if flag not in cmd:
                    cmd.append(flag)
            print(f"\n{'='*90}\n[Suite] 开始模型: {mt}\n命令: {' '.join(cmd)}\n{'='*90}")
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                failed.append({'model_type': mt, 'return_code': rc})
                print(f"[Suite] ⚠️ 模型 {mt} 失败（rc={rc}），继续后续模型")
        failed_path = os.path.join(os.path.abspath(base_output_dir), 'failed_models.json')
        with open(failed_path, 'w', encoding='utf-8') as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
        print(f"[Suite] failed_models -> {failed_path}")
        return failed

    # ---- ablation suite / 自定义消融列表：每个模型独立子进程 ----
    ABLATION_SUITE_MODELS = [
        'yolov8s',
        'yolov8s_ablate_baseline',
        'yolov8s_ablate_adaconv_only',
        'yolov8s_ablate_mscb_only',
        'yolov8s_ablate_pam_only',
        'yolov8s_ablate_rst_only',
        'yolov8s_ablate_mscb_pam',
        'yolov8s_ablate_no_rst',
        'yolov8s_ablate_rst_p5_only',
        'yolov8s_ablate_rst_p4p5',
        'yolov8s_ablate_full',
        'rainforest_paper_v8s',
    ]
    if args.mode == 'train' and args.ablation_suite:
        _run_models_as_subprocess(ABLATION_SUITE_MODELS, args.output_dir)
        summarize_ablation_results(args.output_dir)
        sys.exit(0)
    if args.mode == 'train' and (args.ablation_models or '').strip():
        req_models = [x.strip() for x in args.ablation_models.split(',') if x.strip()]
        unknown = [x for x in req_models if x not in all_train_model_choices]
        if unknown:
            print(f"❌ --ablation_models 含未知模型: {unknown}")
            sys.exit(1)
        _run_models_as_subprocess(req_models, args.output_dir)
        summarize_ablation_results(args.output_dir)
        sys.exit(0)

    # ---- all 模式：顺序跑 baseline + full ----
    ALL_MODEL_TYPES = ['yolov8s', 'rainforest_paper_v8s']
    if args.model_type == 'all' and args.mode == 'train':
        print("🚀 all 模式：将依次测试 YOLOv8s baseline 与 Rainforest-YOLO(v8s)，仅运行 ordered 数据版本")
        incomplete_models, completed_models = _infer_incomplete_models(ALL_MODEL_TYPES, args.output_dir)
        if completed_models and not args.rerun_completed:
            print(f"[ALL] 已完成，自动跳过: {completed_models}")
        target_models = ALL_MODEL_TYPES if args.rerun_completed else incomplete_models
        if not target_models:
            print("[ALL] 没有需要补跑的模型；已全部完成。")
            sys.exit(0)
        print(f"[ALL] 本次将运行: {target_models}")
        _run_models_as_subprocess(target_models, args.output_dir)
        print("\n✅ all 模式执行完成")
        sys.exit(0)

    # ---- eval_video 模式 ----
    if args.mode == 'eval_video':
        run_eval_video(args)
        sys.exit(0)

    # ---- XML→YOLO 导出后退出 ----
    if getattr(args, 'export_yolo', ''):
        export_xml_to_yolo(args.export_yolo)
        print(f"下一步: python augment_rainforest_offline.py --img_dir {args.export_yolo}/images --lbl_dir {args.export_yolo}/labels --out_dir data_augmented --n_aug 5")
        print(f"然后设置 raw_yolo_base='./data_augmented'，运行: python train_k.py --model_type rainforest")
        print(f"一键执行 1+2 步: .\\run_augment_then_train.ps1")
        sys.exit(0)

    # ---- CUDA 初始化（让 nsys 能捕获 CUDA 事件）----
    cuda_available = False
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"[CUDA] CUDA 可用，设备: {torch.cuda.get_device_name(0)}")
            _ = torch.empty(1, device="cuda")
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push("script_start")
            print("[NVTX] CUDA context 已初始化，NVTX 标记已启用")
        else:
            print("[CUDA] ⚠️  警告：CUDA 不可用，NVTX 标记将被跳过")
    except Exception as e:
        print(f"[CUDA] ⚠️  警告：CUDA 初始化失败: {e}")
        cuda_available = False

    model_type = args.model_type

    if model_type in GIBBON_QUALITY_LOSS_TYPES:
        os.environ['RF_GIBBON_QUALITY_LOSS'] = '1'
        os.environ.setdefault('RF_GQL_LAMBDA_NWD', '0.10')
        os.environ.setdefault('RF_GQL_NWD_C', '20.0')
        os.environ.setdefault('RF_GQL_SMALL_TAU', '0.05')
        os.environ.setdefault('RF_GQL_RAMP_START', '5')
        os.environ.setdefault('RF_GQL_RAMP_END', '20')
        print("[GibbonQualityLoss] enabled: lambda_nwd=0.10, C=20.0, small_tau=0.05, ramp=5-20")
    else:
        os.environ.pop('RF_GIBBON_QUALITY_LOSS', None)

    if model_type in ('rainforest_paper_v8s',) or is_yolov8s_ablation_model(model_type) or is_yolov8s_pt_yaml_model(model_type):
        import ultralytics_rainforest  # noqa: F401
    if model_type == 'rainforest_staged':
        if not args.single_fold:
            args.output_dir = results_path_rainforest
        split_yamls = prepare_kfold_data()
        run_staged_training(args, yaml_files=split_yamls)
        sys.exit(0)

    if args.single_fold:
        run_results_path = os.path.abspath(args.output_dir)
        print(f"📁 单划分模式：每个数据源仅使用 1 份 6:3:1 划分，结果保存到 {run_results_path}")
    else:
        if model_type in ('yolo11n', 'yolo11s', 'cbam'):
            run_results_path = os.path.abspath(results_path)
        else:
            run_results_path = os.path.abspath(results_path_rainforest)  # rainforest / n_lite / n_lite_plus / rainforest_arch

    nvtx_range_push("data_preparation")
    data_yamls = prepare_kfold_data()
    nvtx_range_pop()

    print("✅ 数据划分完成：当前实验仅使用 ordered 数据，按 closec/closen/farc/farn 严格均衡后做 1 份 6:3:1（train/val/test）划分")
    print(f"   架构: {model_type}，数据源数: {len(data_yamls)}，结果目录: {run_results_path}")

    # 训练前标签自检
    all_bad = []
    for yf in data_yamls:
        dataset_dir = os.path.dirname(yf)
        for sub in ('train', 'val', 'test'):
            lbl_dir = os.path.join(dataset_dir, sub, 'labels')
            all_bad.extend(check_yolo_labels(lbl_dir))
    if all_bad:
        print(f"\n❌ 标签自检发现 {len(all_bad)} 条异常，请修复后再训练（避免 NaN）。")
        print("   前 20 条:", all_bad[:20])
        sys.exit(1)
    print("✅ 标签自检通过（无越界/无效宽高/NaN）。")

    _apply_wiou_loss_patch()

    # ── 按模式构建 active_params（真正做到每个模型参数独立）──────────────────────
    active_params, cfg_group = resolve_model_active_params(model_type)
    print(f"[Config] {model_type} 使用独立配置组: {cfg_group}")

    if model_type == 'rainforest_paper_v8s':
        active_params['amp'] = False
        print("[AMP-SAFE] rainforest_paper_v8s 已强制 amp=False（避免 RST/LayerNorm 在验证期 Half/Float 冲突）")
    if is_yolov8s_ablation_model(model_type):
        has_rst = ablation_model_has_rst(model_type)
        active_params['amp'] = False if has_rst else True
        print(f"[AMP-SAFE] {model_type} has_rst={has_rst} -> amp={active_params['amp']}")
    if is_yolov8s_pt_yaml_model(model_type):
        active_params['amp'] = True
        print(f"[YOLOv8s+custom-YAML] {model_type} amp=True（无 RST）")

    # ── CLI 参数覆盖（最高优先级）────────────────────────────────────────
    _cli_map = dict(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, nbs=args.nbs, device=args.device, mosaic=args.mosaic,
                    mixup=args.mixup, close_mosaic=args.close_mosaic)
    for k, v in _cli_map.items():
        if v is not None:
            active_params[k] = v
            print(f"[CLI] {k} = {v}")

    # ── 循环训练 ─────────────────────────────────────────────────────────
    n_folds_run = len(data_yamls)
    for i, yaml_file in enumerate(data_yamls):
        fold_num = i + 1
        dataset_name = _extract_dataset_name_from_yaml(yaml_file)
        print(f"\n🚀 开始训练数据源 {dataset_name} ({fold_num}/{n_folds_run}) ...")

        nvtx_range_push(f"fold_{fold_num}_total")

        # 清理旧缓存，防止 Fold 之间数据混淆
        nvtx_range_push(f"fold_{fold_num}_cleanup")
        dataset_dir = os.path.dirname(yaml_file)
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.cache'):
                    os.remove(os.path.join(root, file))
        nvtx_range_pop()

        nvtx_range_push(f"fold_{fold_num}_model_load")
        pretrain_transfer_report = None
        yaml_path_for_log = ''
        if model_type == 'yolov8s':
            model = YOLO('yolov8s.pt')
            yaml_path_for_log = 'yolov8s.pt'
        elif is_yolov8s_ablation_model(model_type):
            meta = ABLATION_MODEL_METADATA[model_type]
            yaml_path_for_log = _resolve_local_path(meta['yaml'])
            model = load_yolov8s_ablation_with_pretrained(meta['yaml'], 'yolov8s.pt', model_name=model_type)
            pretrain_transfer_report = getattr(model, '_pretrain_transfer_report', None)
        elif model_type == 'yolov8s_p2_bifpn_lite':
            import ultralytics_rainforest  # noqa: F401
            yaml_path_for_log = _resolve_local_path('cfg/yolov8s-p2-bifpn-lite.yaml')
            model = load_yolov8s_p2_bifpn_lite_with_pretrained(
                'cfg/yolov8s-p2-bifpn-lite.yaml', 'yolov8s.pt', model_name=model_type
            )
            pretrain_transfer_report = getattr(model, '_pretrain_transfer_report', None)
        elif is_yolov8s_pt_yaml_model(model_type):
            meta = YOLOV8S_PT_YAML_MODEL_METADATA[model_type]
            yaml_path_for_log = _resolve_local_path(meta['yaml'])
            model = load_yolov8s_ablation_with_pretrained(meta['yaml'], 'yolov8s.pt', model_name=model_type)
            pretrain_transfer_report = getattr(model, '_pretrain_transfer_report', None)
        elif model_type == 'rainforest_paper_v8s':
            model = load_rainforest_v8s_paper_with_pretrained('cfg/yolov8s-rainforest-paper.yaml', 'yolov8s.pt')
            yaml_path_for_log = _resolve_local_path('cfg/yolov8s-rainforest-paper.yaml')

        if model_type == 'rainforest_paper_v8s':
            has_rst_model = True
        elif is_yolov8s_ablation_model(model_type):
            has_rst_model = ablation_model_has_rst(model_type)
        elif is_yolov8s_pt_yaml_model(model_type):
            has_rst_model = False
        else:
            has_rst_model = False
        print(f"[ModelInit] model_type={model_type}")
        print(f"[ModelInit] yaml_path={yaml_path_for_log}")
        print(f"[ModelInit] has_rst={has_rst_model}, amp={active_params.get('amp')}")
        if pretrain_transfer_report:
            print(f"[ModelInit] matched/random_init={pretrain_transfer_report.get('matched')}/{pretrain_transfer_report.get('random_init')}")
        params_snapshot_now = {k: active_params.get(k) for k in ('epochs', 'imgsz', 'batch', 'lr0', 'lrf', 'amp', 'mosaic', 'mixup', 'copy_paste', 'close_mosaic')}
        print(f"[ModelInit] params_snapshot={params_snapshot_now}")
        if pretrain_transfer_report and fold_num == 1:
            try:
                with open(os.path.join(run_results_path, 'pretrain_transfer_report.json'), 'w', encoding='utf-8') as f:
                    json.dump(pretrain_transfer_report, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] 根目录 pretrain_transfer_report.json 写入失败: {e}")

        if model_type in ('rainforest_paper_v8s',) or (is_yolov8s_ablation_model(model_type) and ablation_model_has_rst(model_type)) or is_yolov8s_pt_yaml_model(model_type):
            check_ema_compatibility(model.model)
        if cuda_available:
            try:
                if hasattr(model, 'model') and hasattr(model.model, 'to'):
                    model.model.to('cuda')
                _ = torch.empty(1, device="cuda")
                torch.cuda.synchronize()
                print(f"[CUDA] 模型已加载，CUDA context 活跃，设备: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"[CUDA] ⚠️  警告：模型 CUDA 初始化失败: {e}")
        nvtx_range_pop()

        # ── Live Profiler / 训练回调（baseline 默认不挂任何额外训练干预）──────────
        close_profiler = None
        tb_dir = (getattr(args, "tb_logdir", "") or "").strip() or os.path.join(run_results_path, f"tb_fold{fold_num}")
        tb_dir = os.path.abspath(tb_dir)

        if is_custom_rainforest_model(model_type):
            install_grad_clip_callback(model, max_norm=10.0)

        if should_enable_live_profiler(args):
            close_profiler = install_live_profiler_ultralytics(
                model,
                logdir=tb_dir,
                enable_nvtx_batch=bool(getattr(args, "nvtx_batch", False)),
                enable_profiler=bool(getattr(args, "torch_profile", False)),
                prof_wait=30,
                prof_active=10,
                enable_grad_clip=False,
            )
            tb_parent = os.path.dirname(tb_dir)
            print(f"[Profiler] TensorBoard 日志目录: {tb_dir}")
            print("")
            print("  ---------- 打开 TensorBoard 方法 ----------")
            print(f"  1. 新开一个终端，执行：")
            print(f'     python -m tensorboard.main --logdir "{tb_parent}" --port 6006')
            print(f"  2. 浏览器打开: http://localhost:6006")
            print(f"  或直接运行脚本: .\\start_tensorboard_now.ps1")
            print("  --------------------------------------------")
            print("")
            print(f"[Profiler] ⚠️  需等待训练跑几个 batch 后，TensorBoard 才会显示数据")
        else:
            print("[Profiler] 未启用；保持训练路径纯净（无额外 profiler 回调）")

        # ── 构建本折 train_kw（基础参数 + 本折标识）────────────────────
        # baseline 用 active_params 的损失权重；rainforest / rainforest_n_lite 用对应 phase1 的损失权重
        _p1 = CFG_RAINFOREST_N_LITE['phase1'] if model_type in ('rainforest_n', 'rainforest_v8n', 'rainforest_n_lite', 'rainforest_n_lite_plus') else CFG_RAINFOREST['phase1']
        _use_rf_loss = model_type in ('rainforest', 'rainforest_n', 'rainforest_v8n', 'rainforest_n_lite', 'rainforest_n_lite_plus')
        _box = args.box if args.box is not None else (_p1['box'] if _use_rf_loss else active_params['box'])
        _dfl = args.dfl if args.dfl is not None else (_p1['dfl'] if _use_rf_loss else active_params['dfl'])
        _cls = args.cls if args.cls is not None else (_p1['cls'] if _use_rf_loss else active_params['cls'])
        _iou = _p1['iou'] if _use_rf_loss else active_params.get('iou', 0.65)
        train_kw = {
            **active_params,
            'data': yaml_file,
            'project': run_results_path,
            'name': f"fold_{fold_num}",
            'plots': True,
            'seed': 42 + fold_num,
            'deterministic': False,
            'box': _box,
            'dfl': _dfl,
            'cls': _cls,
            'iou': _iou,
        }

        # ── rainforest 专属设置 ──────────────────────────────────────────
        if model_type == 'rainforest':
            train_kw['pretrained'] = False
            train_kw['amp'] = False

            # 清理不兼容旧目录（层数不等于 31 的历史权重目录）
            import re
            _stale_pattern = re.compile(rf'^fold_{fold_num}_phase[12]\d*$')
            os.makedirs(run_results_path, exist_ok=True)
            for _d in os.listdir(run_results_path):
                if _stale_pattern.match(_d):
                    _full = os.path.join(run_results_path, _d)
                    if os.path.isdir(_full):
                        _w = os.path.join(_full, 'weights', 'best.pt')
                        _is_stale = True
                        if os.path.isfile(_w):
                            try:
                                _ck = torch.load(_w, map_location='cpu', weights_only=False)
                                _n_layers = len(_ck.get('model', {}).model) if hasattr(_ck.get('model', {}), 'model') else 0
                                _is_stale = (_n_layers > 0 and _n_layers != 31)
                            except Exception:
                                _is_stale = True
                        if _is_stale:
                            print(f"  [Rainforest] 🗑  删除不兼容旧目录: {_d}")
                            shutil.rmtree(_full, ignore_errors=True)

            # ── Phase 1：冻结纯继承层 ────────────────────────────────────
            _p2 = CFG_RAINFOREST['phase2']
            phase1_freeze = get_rainforest_phase1_freeze_indices(model)
            print(f"\n  [Rainforest] Phase 1: 冻结 {phase1_freeze}（{len(phase1_freeze)} 层）, {RAINFOREST_FREEZE_EP} ep，弱增强")

            train_kw_p1 = {
                **train_kw,
                'epochs'      : RAINFOREST_FREEZE_EP,
                'freeze'      : phase1_freeze,
                'name'        : f'fold_{fold_num}_phase1',
                # Phase1 专属增强（来自 CFG_RAINFOREST.phase1）
                'mosaic'      : _p1['mosaic'],
                'close_mosaic': _p1['close_mosaic'],
                'mixup'       : _p1['mixup'],
                'copy_paste'  : _p1['copy_paste'],
                'dropout'     : _p1['dropout'],
                'multi_scale' : _p1['multi_scale'],
            }
            nvtx_range_push(f"fold_{fold_num}_train")
            model.train(**train_kw_p1)
            nvtx_range_pop()

            p1_save_dir = str(model.trainer.save_dir)
            best_pt_p1  = os.path.join(p1_save_dir, 'weights', 'best.pt')
            if not os.path.isfile(best_pt_p1):
                best_pt_p1 = os.path.join(p1_save_dir, 'weights', 'last.pt')
            print(f"  [Rainforest] Phase 1 保存目录: {p1_save_dir}")

            if os.path.isfile(best_pt_p1):
                print(f"\n  [Rainforest] Phase 2: 全解冻, {RAINFOREST_UNFREEZE_EP} ep（从 Phase1 best），精修增强")
                model = YOLO(best_pt_p1)
                if close_profiler is not None:
                    close_profiler()
                close_profiler = None
                if should_enable_live_profiler(args):
                    close_profiler = install_live_profiler_ultralytics(
                        model, logdir=tb_dir,
                        enable_nvtx_batch=bool(getattr(args, "nvtx_batch", False)),
                        enable_profiler=bool(getattr(args, "torch_profile", False)),
                        prof_wait=5, prof_active=10,
                        enable_grad_clip=False,
                    )
                # Phase2 覆盖（来自 CFG_RAINFOREST.phase2，未列出项继承 phase1）
                train_kw_p2 = {
                    **train_kw,
                    'epochs'        : RAINFOREST_UNFREEZE_EP,
                    'name'          : f'fold_{fold_num}_phase2',
                    'warmup_epochs' : _p2['warmup_epochs'],
                    'amp'           : _p2['amp'],
                    'mosaic'        : _p2['mosaic'],
                    'close_mosaic'  : _p2['close_mosaic'],
                    'multi_scale'   : _p2['multi_scale'],
                }
                train_kw_p2.pop('freeze', None)
                nvtx_range_push(f"fold_{fold_num}_train")
                model.train(**train_kw_p2)
                nvtx_range_pop()
                fold_save_dir = str(model.trainer.save_dir)
            else:
                print(f"  [Rainforest] ⚠ Phase1 best.pt 未找到，跳过 Phase2")
                fold_save_dir = p1_save_dir

        elif model_type == 'rainforest_paper_v8s':
            # 三阶段“控制增强 + 分阶段学习”
            phased = CFG_RAINFOREST_V8S_PAPER_PHASED
            shared = phased['shared']
            print(f"\n  [rainforest_paper_v8s] 三阶段训练: P1({phased['phase1']['epochs']}) + P2({phased['phase2']['epochs']}) + P3({phased['phase3']['epochs']}) = 320 ep")
            print("  [rainforest_paper_v8s] 已禁用冲突增强：mixup=0 / copy_paste=0 / auto_augment=None / erasing=0")

            phase_plan = [
                ("phase1", phased['phase1'], None),
                ("phase2", phased['phase2'], None),
                ("phase3", phased['phase3'], phased['phase3'].get('lr0_scale', 0.1)),
            ]
            prev_best = None
            fold_save_dir = None

            for stage_idx, (stage_name, stage_cfg, lr_scale) in enumerate(phase_plan, start=1):
                if prev_best and os.path.isfile(prev_best):
                    model = YOLO(prev_best)
                    if close_profiler is not None:
                        close_profiler()
                    close_profiler = None
                    if should_enable_live_profiler(args):
                        close_profiler = install_live_profiler_ultralytics(
                            model, logdir=tb_dir,
                            enable_nvtx_batch=bool(getattr(args, "nvtx_batch", False)),
                            enable_profiler=bool(getattr(args, "torch_profile", False)),
                            prof_wait=5, prof_active=10,
                            enable_grad_clip=False,
                        )

                stage_lr0 = shared['lr0'] * lr_scale if lr_scale is not None else shared['lr0']
                stage_kw = {
                    **train_kw,
                    'name': f'fold_{fold_num}_{stage_name}',
                    'epochs': stage_cfg['epochs'],
                    'imgsz': shared['imgsz'],
                    'batch': shared['batch'],
                    'nbs': shared.get('nbs', 8),
                    'device': shared.get('device', 0),
                    'optimizer': shared['optimizer'],
                    'lr0': stage_lr0,
                    'lrf': shared['lrf'],
                    'amp': shared['amp'],
                    'warmup_epochs': stage_cfg['warmup_epochs'],
                    'patience': stage_cfg['patience'],
                    'mosaic': stage_cfg['mosaic'],
                    'close_mosaic': stage_cfg['close_mosaic'],
                    'multi_scale': stage_cfg['multi_scale'],
                    'mixup': shared['mixup'],
                    'copy_paste': shared['copy_paste'],
                    'auto_augment': shared['auto_augment'],
                    'erasing': shared['erasing'],
                }
                stage_kw.pop('freeze', None)

                # 最终强制覆盖：避免分阶段默认配置覆盖 CLI 参数
                if getattr(args, "imgsz", None) is not None:
                    stage_kw["imgsz"] = int(args.imgsz)
                if getattr(args, "batch", None) is not None:
                    stage_kw["batch"] = int(args.batch)
                if getattr(args, "nbs", None) is not None:
                    stage_kw["nbs"] = int(args.nbs)
                if getattr(args, "device", None) is not None:
                    stage_kw["device"] = str(args.device)

                print("[FORCE OVERRIDE] final stage_kw imgsz =", stage_kw.get("imgsz"))
                print("[FORCE OVERRIDE] final stage_kw batch =", stage_kw.get("batch"))
                print("[FORCE OVERRIDE] final stage_kw nbs   =", stage_kw.get("nbs"))
                print("[FORCE OVERRIDE] final stage_kw device=", stage_kw.get("device"))

                print(
                    f"  [rainforest_paper_v8s] Stage {stage_idx}/3 {stage_name}: "
                    f"epochs={stage_cfg['epochs']} lr0={stage_lr0:.6g} "
                    f"mosaic={stage_cfg['mosaic']} multi_scale={stage_cfg['multi_scale']}"
                )
                nvtx_range_push(f"fold_{fold_num}_train_{stage_name}")
                model.train(**stage_kw)
                nvtx_range_pop()

                fold_save_dir = str(model.trainer.save_dir)
                prev_best = os.path.join(fold_save_dir, 'weights', 'best.pt')
                if not os.path.isfile(prev_best):
                    prev_best = os.path.join(fold_save_dir, 'weights', 'last.pt')
                if not os.path.isfile(prev_best):
                    print(f"  [rainforest_paper_v8s] ⚠ {stage_name} 未找到可续训权重，提前结束分阶段训练")
                    break

            if fold_save_dir is None:
                fold_save_dir = str(model.trainer.save_dir)

        elif is_monkeyplus_phased_model(model_type):
            phased = CFG_MONKEYPLUS_PHASED
            shared = phased['shared']
            p1, p2 = phased['phase1'], phased['phase2']
            print(f"\n  [MonkeyPlus] 二阶段: phase1 {p1['epochs']}ep @{p1['imgsz']} + phase2 {p2['epochs']}ep @{p2['imgsz']} 定位精修")
            phase_plan = [
                ("phase1", p1, None),
                ("phase2", p2, p2.get('lr0_scale', 0.1)),
            ]
            prev_best = None
            fold_save_dir = None
            base_lr0 = float(train_kw['lr0'])

            for stage_idx, (stage_name, stage_cfg, lr_scale) in enumerate(phase_plan, start=1):
                if prev_best and os.path.isfile(prev_best):
                    model = YOLO(prev_best)
                    if close_profiler is not None:
                        close_profiler()
                    close_profiler = None
                    if should_enable_live_profiler(args):
                        close_profiler = install_live_profiler_ultralytics(
                            model, logdir=tb_dir,
                            enable_nvtx_batch=bool(getattr(args, "nvtx_batch", False)),
                            enable_profiler=bool(getattr(args, "torch_profile", False)),
                            prof_wait=5, prof_active=10,
                            enable_grad_clip=False,
                        )

                stage_lr0 = base_lr0 * lr_scale if lr_scale is not None else base_lr0
                stage_kw = {
                    **train_kw,
                    'name': f'fold_{fold_num}_{stage_name}',
                    'epochs': stage_cfg['epochs'],
                    'imgsz': stage_cfg['imgsz'],
                    'batch': stage_cfg['batch'],
                    'nbs': stage_cfg.get('nbs', train_kw.get('nbs', 64)),
                    'lr0': stage_lr0,
                    'lrf': shared['lrf'],
                    'amp': shared['amp'],
                    'warmup_epochs': stage_cfg['warmup_epochs'],
                    'patience': stage_cfg['patience'],
                    'mosaic': stage_cfg['mosaic'],
                    'close_mosaic': stage_cfg['close_mosaic'],
                    'multi_scale': stage_cfg['multi_scale'],
                    'mixup': shared['mixup'],
                    'copy_paste': stage_cfg.get('copy_paste', 0.0),
                    'optimizer': train_kw.get('optimizer', 'auto'),
                    'auto_augment': shared['auto_augment'],
                    'erasing': shared['erasing'],
                }
                stage_kw.pop('freeze', None)

                if stage_idx == 1:
                    if getattr(args, "imgsz", None) is not None:
                        stage_kw["imgsz"] = int(args.imgsz)
                    if getattr(args, "batch", None) is not None:
                        stage_kw["batch"] = int(args.batch)
                if getattr(args, "nbs", None) is not None:
                    stage_kw["nbs"] = int(args.nbs)
                if getattr(args, "device", None) is not None:
                    stage_kw["device"] = str(args.device)

                print(
                    f"  [MonkeyPlus] Stage {stage_idx}/2 {stage_name}: "
                    f"epochs={stage_cfg['epochs']} imgsz={stage_kw['imgsz']} batch={stage_kw['batch']} lr0={stage_lr0:.6g}"
                )
                nvtx_range_push(f"fold_{fold_num}_train_{stage_name}")
                model.train(**stage_kw)
                nvtx_range_pop()

                fold_save_dir = str(model.trainer.save_dir)
                prev_best = os.path.join(fold_save_dir, 'weights', 'best.pt')
                if not os.path.isfile(prev_best):
                    prev_best = os.path.join(fold_save_dir, 'weights', 'last.pt')
                if not os.path.isfile(prev_best):
                    print(f"  [MonkeyPlus] ⚠ {stage_name} 未找到可续训权重，提前结束二阶段训练")
                    break

            if fold_save_dir is None:
                fold_save_dir = str(model.trainer.save_dir)

        elif model_type == 'rainforest_n_lite':
            train_kw['pretrained'] = False
            train_kw['amp'] = False

            # 清理不兼容旧目录（层数不等于 27 的 n-lite 历史目录）
            import re
            _stale_pattern_nlite = re.compile(rf'^fold_{fold_num}_phase[12]\d*$')
            os.makedirs(run_results_path, exist_ok=True)
            for _d in os.listdir(run_results_path):
                if _stale_pattern_nlite.match(_d):
                    _full = os.path.join(run_results_path, _d)
                    if os.path.isdir(_full):
                        _w = os.path.join(_full, 'weights', 'best.pt')
                        _is_stale = True
                        if os.path.isfile(_w):
                            try:
                                _ck = torch.load(_w, map_location='cpu', weights_only=False)
                                _md = _ck.get('model', {})
                                _n_layers = len(getattr(_md, 'model', [])) if hasattr(_md, 'model') else 0
                                _is_stale = (_n_layers > 0 and _n_layers != 27)
                            except Exception:
                                _is_stale = True
                        if _is_stale:
                            print(f"  [n-lite] 🗑  删除不兼容旧目录: {_d}")
                            shutil.rmtree(_full, ignore_errors=True)

            # ── Phase 1：冻结 MSCB + PAM ───────────────────────────────────
            module_map = discover_module_indices(model)
            phase1_freeze = get_freeze_indices_staged(module_map, ['MSCB', 'PAM'])
            _np1 = CFG_RAINFOREST_N_LITE['phase1']
            print(f"\n  [n-lite] Phase 1: 冻结 {phase1_freeze}（MSCB+PAM）, {N_LITE_FREEZE_EP} ep")

            train_kw_p1 = {
                **train_kw,
                'epochs': N_LITE_FREEZE_EP,
                'freeze': phase1_freeze,
                'name': f'fold_{fold_num}_phase1',
                'mosaic': _np1['mosaic'],
                'close_mosaic': _np1['close_mosaic'],
                'mixup': _np1['mixup'],
                'copy_paste': _np1['copy_paste'],
                'dropout': _np1['dropout'],
                'multi_scale': _np1['multi_scale'],
            }
            nvtx_range_push(f"fold_{fold_num}_train")
            model.train(**train_kw_p1)
            nvtx_range_pop()

            p1_save_dir = str(model.trainer.save_dir)
            best_pt_p1 = os.path.join(p1_save_dir, 'weights', 'best.pt')
            if not os.path.isfile(best_pt_p1):
                best_pt_p1 = os.path.join(p1_save_dir, 'weights', 'last.pt')
            print(f"  [n-lite] Phase 1 保存目录: {p1_save_dir}")

            if os.path.isfile(best_pt_p1):
                _np2 = CFG_RAINFOREST_N_LITE['phase2']
                print(f"\n  [n-lite] Phase 2: 全解冻, {N_LITE_UNFREEZE_EP} ep（从 Phase1 best）")
                model = YOLO(best_pt_p1)
                if close_profiler is not None:
                    close_profiler()
                close_profiler = None
                if should_enable_live_profiler(args):
                    close_profiler = install_live_profiler_ultralytics(
                        model, logdir=tb_dir,
                        enable_nvtx_batch=bool(getattr(args, "nvtx_batch", False)),
                        enable_profiler=bool(getattr(args, "torch_profile", False)),
                        prof_wait=5, prof_active=10,
                        enable_grad_clip=False,
                    )
                train_kw_p2 = {
                    **train_kw,
                    'epochs': N_LITE_UNFREEZE_EP,
                    'name': f'fold_{fold_num}_phase2',
                    'warmup_epochs': _np2['warmup_epochs'],
                    'amp': _np2.get('amp', False),
                    'patience': _np2.get('patience', 50),
                    'mosaic': _np2['mosaic'],
                    'close_mosaic': _np2['close_mosaic'],
                    'mixup': _np2.get('mixup', 0),
                    'copy_paste': _np2.get('copy_paste', 0),
                    'dropout': _np2.get('dropout', 0.1),
                    'multi_scale': _np2['multi_scale'],
                }
                train_kw_p2.pop('freeze', None)
                nvtx_range_push(f"fold_{fold_num}_train")
                model.train(**train_kw_p2)
                nvtx_range_pop()
                fold_save_dir = str(model.trainer.save_dir)
            else:
                print(f"  [n-lite] ⚠ Phase1 best.pt 未找到，跳过 Phase2")
                fold_save_dir = p1_save_dir

        elif model_type == 'rainforest_n_lite_plus':
            train_kw['pretrained'] = False
            train_kw['amp'] = False

            import re
            _stale_pattern_nlite_plus = re.compile(rf'^fold_{fold_num}_phase[12]\d*$')
            os.makedirs(run_results_path, exist_ok=True)
            for _d in os.listdir(run_results_path):
                if _stale_pattern_nlite_plus.match(_d):
                    _full = os.path.join(run_results_path, _d)
                    if os.path.isdir(_full):
                        _w = os.path.join(_full, 'weights', 'best.pt')
                        _is_stale = True
                        if os.path.isfile(_w):
                            try:
                                _ck = torch.load(_w, map_location='cpu', weights_only=False)
                                _md = _ck.get('model', {})
                                _n_layers = len(getattr(_md, 'model', [])) if hasattr(_md, 'model') else 0
                                _is_stale = (_n_layers > 0 and _n_layers != 28)
                            except Exception:
                                _is_stale = True
                        if _is_stale:
                            print(f"  [n-lite+] 🗑  删除不兼容旧目录: {_d}")
                            shutil.rmtree(_full, ignore_errors=True)

            module_map = discover_module_indices(model)
            phase1_freeze = get_freeze_indices_staged(module_map, ['MSCB', 'PAM', 'RST'])
            _np1 = CFG_RAINFOREST_N_LITE['phase1']
            print(f"\n  [n-lite+] Phase 1: 冻结 {phase1_freeze}（MSCB+PAM+P5-RST）, {N_LITE_FREEZE_EP} ep")

            train_kw_p1 = {
                **train_kw,
                'epochs': N_LITE_FREEZE_EP,
                'freeze': phase1_freeze,
                'name': f'fold_{fold_num}_phase1',
                'mosaic': _np1['mosaic'],
                'close_mosaic': _np1['close_mosaic'],
                'mixup': _np1['mixup'],
                'copy_paste': _np1['copy_paste'],
                'dropout': _np1['dropout'],
                'multi_scale': _np1['multi_scale'],
            }
            nvtx_range_push(f"fold_{fold_num}_train")
            model.train(**train_kw_p1)
            nvtx_range_pop()

            p1_save_dir = str(model.trainer.save_dir)
            best_pt_p1 = os.path.join(p1_save_dir, 'weights', 'best.pt')
            if not os.path.isfile(best_pt_p1):
                best_pt_p1 = os.path.join(p1_save_dir, 'weights', 'last.pt')
            print(f"  [n-lite+] Phase 1 保存目录: {p1_save_dir}")

            if os.path.isfile(best_pt_p1):
                _np2 = CFG_RAINFOREST_N_LITE['phase2']
                print(f"\n  [n-lite+] Phase 2: 全解冻, {N_LITE_UNFREEZE_EP} ep（从 Phase1 best）")
                model = YOLO(best_pt_p1)
                if close_profiler is not None:
                    close_profiler()
                close_profiler = None
                if should_enable_live_profiler(args):
                    close_profiler = install_live_profiler_ultralytics(
                        model, logdir=tb_dir,
                        enable_nvtx_batch=bool(getattr(args, "nvtx_batch", False)),
                        enable_profiler=bool(getattr(args, "torch_profile", False)),
                        prof_wait=5, prof_active=10,
                        enable_grad_clip=False,
                    )
                train_kw_p2 = {
                    **train_kw,
                    'epochs': N_LITE_UNFREEZE_EP,
                    'name': f'fold_{fold_num}_phase2',
                    'warmup_epochs': _np2['warmup_epochs'],
                    'amp': _np2.get('amp', False),
                    'patience': _np2.get('patience', 50),
                    'mosaic': _np2['mosaic'],
                    'close_mosaic': _np2['close_mosaic'],
                    'mixup': _np2.get('mixup', 0),
                    'copy_paste': _np2.get('copy_paste', 0),
                    'dropout': _np2.get('dropout', 0.1),
                    'multi_scale': _np2['multi_scale'],
                }
                train_kw_p2.pop('freeze', None)
                nvtx_range_push(f"fold_{fold_num}_train")
                model.train(**train_kw_p2)
                nvtx_range_pop()
                fold_save_dir = str(model.trainer.save_dir)
            else:
                print(f"  [n-lite+] ⚠ Phase1 best.pt 未找到，跳过 Phase2")
                fold_save_dir = p1_save_dir

        elif model_type == 'rainforest_n':
            # ── yolo11n-rainforest 完整版（32 层，4 项全部改进 + 3×RST）────────
            train_kw['pretrained'] = False
            train_kw['amp'] = False

            import re
            _stale_pattern_nfull = re.compile(rf'^fold_{fold_num}_phase[12]\d*$')
            os.makedirs(run_results_path, exist_ok=True)
            for _d in os.listdir(run_results_path):
                if _stale_pattern_nfull.match(_d):
                    _full = os.path.join(run_results_path, _d)
                    if os.path.isdir(_full):
                        _w = os.path.join(_full, 'weights', 'best.pt')
                        _is_stale = True
                        if os.path.isfile(_w):
                            try:
                                _ck = torch.load(_w, map_location='cpu', weights_only=False)
                                _md = _ck.get('model', {})
                                _n_layers = len(getattr(_md, 'model', [])) if hasattr(_md, 'model') else 0
                                _is_stale = (_n_layers > 0 and _n_layers != 32)
                            except Exception:
                                _is_stale = True
                        if _is_stale:
                            print(f"  [n-full] 🗑  删除不兼容旧目录: {_d}")
                            shutil.rmtree(_full, ignore_errors=True)

            module_map = discover_module_indices(model)
            phase1_freeze = get_freeze_indices_staged(module_map, ['MSCB', 'PAM', 'RST'])
            _np1 = CFG_RAINFOREST_N_LITE['phase1']
            print(f"\n  [n-full] Phase 1: 冻结 {phase1_freeze}（MSCB+PAM+RST×3）, {N_LITE_FREEZE_EP} ep")

            train_kw_p1 = {
                **train_kw,
                'epochs': N_LITE_FREEZE_EP,
                'freeze': phase1_freeze,
                'name': f'fold_{fold_num}_phase1',
                'mosaic': _np1['mosaic'],
                'close_mosaic': _np1['close_mosaic'],
                'mixup': _np1['mixup'],
                'copy_paste': _np1['copy_paste'],
                'dropout': _np1['dropout'],
                'multi_scale': _np1['multi_scale'],
            }
            nvtx_range_push(f"fold_{fold_num}_train")
            model.train(**train_kw_p1)
            nvtx_range_pop()

            p1_save_dir = str(model.trainer.save_dir)
            best_pt_p1 = os.path.join(p1_save_dir, 'weights', 'best.pt')
            if not os.path.isfile(best_pt_p1):
                best_pt_p1 = os.path.join(p1_save_dir, 'weights', 'last.pt')
            print(f"  [n-full] Phase 1 保存目录: {p1_save_dir}")

            if os.path.isfile(best_pt_p1):
                _np2 = CFG_RAINFOREST_N_LITE['phase2']
                print(f"\n  [n-full] Phase 2: 全解冻, {N_LITE_UNFREEZE_EP} ep（从 Phase1 best）")
                model = YOLO(best_pt_p1)
                if close_profiler is not None:
                    close_profiler()
                close_profiler = None
                if should_enable_live_profiler(args):
                    close_profiler = install_live_profiler_ultralytics(
                        model, logdir=tb_dir,
                        enable_nvtx_batch=bool(getattr(args, "nvtx_batch", False)),
                        enable_profiler=bool(getattr(args, "torch_profile", False)),
                        prof_wait=5, prof_active=10,
                        enable_grad_clip=False,
                    )
                train_kw_p2 = {
                    **train_kw,
                    'epochs': N_LITE_UNFREEZE_EP,
                    'name': f'fold_{fold_num}_phase2',
                    'warmup_epochs': _np2['warmup_epochs'],
                    'amp': _np2.get('amp', False),
                    'patience': _np2.get('patience', 50),
                    'mosaic': _np2['mosaic'],
                    'close_mosaic': _np2['close_mosaic'],
                    'mixup': _np2.get('mixup', 0),
                    'copy_paste': _np2.get('copy_paste', 0),
                    'dropout': _np2.get('dropout', 0.1),
                    'multi_scale': _np2['multi_scale'],
                }
                train_kw_p2.pop('freeze', None)
                nvtx_range_push(f"fold_{fold_num}_train")
                model.train(**train_kw_p2)
                nvtx_range_pop()
                fold_save_dir = str(model.trainer.save_dir)
            else:
                print(f"  [n-full] ⚠ Phase1 best.pt 未找到，跳过 Phase2")
                fold_save_dir = p1_save_dir

        elif model_type == 'rainforest_v8n':
            # ── YOLOv8n-rainforest（31 层，论文最忠实版）────────
            # AMP + RST LayerNorm 在验证时冲突（训练 ok 但 val 崩溃），暂用 FP32
            # workers=4 仍提供 ~2x 加速
            train_kw['pretrained'] = False
            train_kw['amp'] = False
            train_kw['workers'] = 4     # ★ 多进程数据加载（主要加速来源）

            import re
            _stale_pattern_v8n = re.compile(rf'^fold_{fold_num}_phase[12]\d*$')
            os.makedirs(run_results_path, exist_ok=True)
            for _d in os.listdir(run_results_path):
                if _stale_pattern_v8n.match(_d):
                    _full = os.path.join(run_results_path, _d)
                    if os.path.isdir(_full):
                        _w = os.path.join(_full, 'weights', 'best.pt')
                        _is_stale = True
                        if os.path.isfile(_w):
                            try:
                                _ck = torch.load(_w, map_location='cpu', weights_only=False)
                                _md = _ck.get('model', {})
                                _n_layers = len(getattr(_md, 'model', [])) if hasattr(_md, 'model') else 0
                                _is_stale = (_n_layers > 0 and _n_layers != 31)
                            except Exception:
                                _is_stale = True
                        if _is_stale:
                            print(f"  [v8n-rf] 🗑  删除不兼容旧目录: {_d}")
                            shutil.rmtree(_full, ignore_errors=True)

            module_map = discover_module_indices(model)
            phase1_freeze = get_freeze_indices_staged(module_map, ['MSCB', 'PAM', 'RST'])
            _np1 = CFG_RAINFOREST_N_LITE['phase1']
            print(f"\n  [v8n-rf] Phase 1: 冻结 {phase1_freeze}（MSCB+PAM+RST×3）, {N_LITE_FREEZE_EP} ep, workers=4")

            train_kw_p1 = {
                **train_kw,
                'epochs': N_LITE_FREEZE_EP,
                'freeze': phase1_freeze,
                'name': f'fold_{fold_num}_phase1',
                'mosaic': _np1['mosaic'],
                'close_mosaic': _np1['close_mosaic'],
                'mixup': _np1['mixup'],
                'copy_paste': _np1['copy_paste'],
                'dropout': _np1['dropout'],
                'multi_scale': _np1['multi_scale'],
            }
            nvtx_range_push(f"fold_{fold_num}_train")
            model.train(**train_kw_p1)
            nvtx_range_pop()

            p1_save_dir = str(model.trainer.save_dir)
            best_pt_p1 = os.path.join(p1_save_dir, 'weights', 'best.pt')
            if not os.path.isfile(best_pt_p1):
                best_pt_p1 = os.path.join(p1_save_dir, 'weights', 'last.pt')
            print(f"  [v8n-rf] Phase 1 保存目录: {p1_save_dir}")

            if os.path.isfile(best_pt_p1):
                _np2 = CFG_RAINFOREST_N_LITE['phase2']
                print(f"\n  [v8n-rf] Phase 2: 全解冻, {N_LITE_UNFREEZE_EP} ep, workers=4, 强增强")
                model = YOLO(best_pt_p1)
                if close_profiler is not None:
                    close_profiler()
                close_profiler = None
                if should_enable_live_profiler(args):
                    close_profiler = install_live_profiler_ultralytics(
                        model, logdir=tb_dir,
                        enable_nvtx_batch=bool(getattr(args, "nvtx_batch", False)),
                        enable_profiler=bool(getattr(args, "torch_profile", False)),
                        prof_wait=5, prof_active=10,
                        enable_grad_clip=False,
                    )
                train_kw_p2 = {
                    **train_kw,
                    'epochs': N_LITE_UNFREEZE_EP,
                    'name': f'fold_{fold_num}_phase2',
                    'warmup_epochs': _np2['warmup_epochs'],
                    'amp': False,       # RST LayerNorm 与 AMP 验证冲突
                    'workers': 4,       # ★ 多进程加载
                    'mosaic': _np2['mosaic'],
                    'close_mosaic': _np2['close_mosaic'],
                    'mixup': _np2.get('mixup', 0),
                    'copy_paste': _np2.get('copy_paste', 0),
                    'dropout': _np2.get('dropout', 0.1),
                    'multi_scale': _np2['multi_scale'],
                }
                train_kw_p2.pop('freeze', None)
                nvtx_range_push(f"fold_{fold_num}_train")
                model.train(**train_kw_p2)
                nvtx_range_pop()
                fold_save_dir = str(model.trainer.save_dir)
            else:
                print(f"  [v8n-rf] ⚠ Phase1 best.pt 未找到，跳过 Phase2")
                fold_save_dir = p1_save_dir

        else:
            nvtx_range_push(f"fold_{fold_num}_train")
            model.train(**train_kw)
            nvtx_range_pop()
            fold_save_dir = str(model.trainer.save_dir)

        # ── 可选：无强增强精修阶段（专攻 mAP50-95）───────────────────────
        best_pt = os.path.join(fold_save_dir, "weights", "best.pt")
        _ft_cfg = CFG_P2_FINETUNE if is_p2_track_model(model_type) else CFG_RAINFOREST['finetune']
        finetune_epochs = args.finetune_epochs if args.finetune_epochs is not None else _ft_cfg.get('finetune_epochs', 0)
        if model_type == 'rainforest_paper_v8s':
            finetune_epochs = 0
        if is_yolov8s_ablation_model(model_type):
            finetune_epochs = 0
        if is_monkeyplus_phased_model(model_type):
            finetune_epochs = 0
        elif is_p2_track_model(model_type):
            if args.finetune_epochs is not None:
                finetune_epochs = args.finetune_epochs
            else:
                finetune_epochs = P2_MODEL_DEFAULT_FINETUNE_EPOCHS.get(model_type, 0)
        ran_finetune = finetune_epochs > 0 and os.path.isfile(best_pt)
        if ran_finetune:
            print(f"\n  ── 精修阶段: {finetune_epochs} ep @imgsz800 (mosaic=0, lr×{_ft_cfg['lr0_scale']}) ──")
            ft = YOLO(best_pt)
            ft_kw = {
                **active_params,
                **_ft_cfg,           # mosaic/mixup/copy_paste/erasing/multi_scale
                'data'    : yaml_file,
                'project' : run_results_path,
                'name'    : f"{dataset_name}_finetune",
                'epochs'  : finetune_epochs,
                'lr0'     : active_params['lr0'] * _ft_cfg['lr0_scale'],
                'lrf'     : active_params.get('lrf', 0.01),
                'plots'   : True,
                'seed'    : 42 + fold_num,
                'box'     : train_kw['box'],
                'dfl'     : train_kw['dfl'],
                'cls'     : train_kw['cls'],
                'iou'     : train_kw['iou'],
            }
            ft_kw.pop('finetune_epochs', None)
            ft_kw.pop('lr0_scale', None)
            if is_p2_track_model(model_type):
                ft_kw['imgsz'] = 800
                _b = int(ft_kw.get('batch', active_params.get('batch', 8)) or 8)
                ft_kw['batch'] = min(_b, 4)
                ft_kw['amp'] = active_params.get('amp', True)
                print(f"  [P2 finetune] imgsz=800 batch={ft_kw['batch']} lr0={ft_kw['lr0']:.6g}")
            if model_type in ('rainforest_paper_v8s',):
                ft_kw['amp'] = False
            ft.train(**ft_kw)
            fold_save_dir = str(ft.trainer.save_dir)
            del ft
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── 实验记录 ────────────────────────────────────────────────────
        csv_path = os.path.join(fold_save_dir, "results.csv")
        if pretrain_transfer_report and os.path.isdir(fold_save_dir):
            try:
                with open(os.path.join(fold_save_dir, "pretrain_transfer_report.json"), 'w', encoding='utf-8') as f:
                    json.dump(pretrain_transfer_report, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] pretrain_transfer_report 保存失败: {e}")
        val_metrics = _read_val_metrics_from_csv(csv_path)
        best_weights = os.path.join(fold_save_dir, "weights", "best.pt")
        if not os.path.isfile(best_weights):
            last_weights = os.path.join(fold_save_dir, "weights", "last.pt")
            best_weights = last_weights if os.path.isfile(last_weights) else ''
        test_metrics = evaluate_test_split(best_weights, yaml_file, run_results_path, dataset_name, model_type) if best_weights else None
        git_commit, git_diff_stat = _git_commit_and_diff()
        record = {
            "timestamp"      : time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_type"     : model_type,
            "dataset_name"   : dataset_name,
            "fold"           : fold_num,
            "params_snapshot": {k: active_params.get(k) for k in
                                ("epochs", "imgsz", "batch", "lr0", "lrf", "amp", "mosaic", "mixup", "copy_paste", "close_mosaic")},
            "loss_weights"   : {"box": train_kw["box"], "dfl": train_kw["dfl"], "cls": train_kw["cls"], "iou": train_kw["iou"]},
            "finetune_epochs": finetune_epochs if ran_finetune else 0,
            "val_metrics"    : val_metrics,
            "test_metrics"   : test_metrics,
            "conf_iou_note"  : "val 来自训练期；test 来自训练完成后 model.val(split='test')；推理阈值仍可单独调",
            "git_commit"     : git_commit,
            "git_diff_stat"  : git_diff_stat,
            "pretrain_transfer_report": pretrain_transfer_report,
            "yaml_path": yaml_path_for_log,
        }
        append_experiment_log(run_results_path, record)
        print(f"[实验记录] 已追加到 {run_results_path}/experiment_log.json")

        if close_profiler is not None:
            close_profiler()
            close_profiler = None

        # 换折前释放显存
        nvtx_range_push(f"fold_{fold_num}_cleanup_memory")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(5)
        nvtx_range_pop()

        nvtx_range_pop()  # fold_{fold_num}_total

    print(f"\n🎉 所有训练已完成！结果保存在: {run_results_path}")
    print(f"   实验记录: {run_results_path}/experiment_log.json")
    print(f"   对比误检: python train_k.py --mode eval_video --weights <path/to/best.pt> --source <视频> --conf 0.35 --iou 0.6")

    if cuda_available:
        try:
            torch.cuda.nvtx.range_pop()  # script_start
        except Exception:
            pass