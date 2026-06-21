"""
高分辨率定位精修（不重训）：以已训好的 best.pt 为初始化权重，开一个全新的
50ep@800 精修实验（不 resume，避免旧优化器状态/旧 schedule/旧增强干扰），
专攻 mAP50-95（框边界质量 / 高 IoU 稳定性）。

关键点：
  - optimizer 显式设为 AdamW —— 否则 optimizer=auto 会忽略 lr0，本次精修就白调了。
  - 精修后同时做 test@640（与现有排名同口径）和 test@800（精修后最佳推理分辨率），
    以区分“提升来自模型”还是“来自评估分辨率”。

用法（在 yolo5060 环境下）：
  python ft800_from_best.py \
      --best result_p2_respam_sgf/fold_1/weights/best.pt \
      --project result_p2_respam_sgf_ft800
"""
import argparse
import json
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "ultralytics_local"))
import ultralytics_rainforest  # noqa: F401  注册 ResPAM/SGF 等自定义模块（加载 best.pt 必需）
from ultralytics import YOLO

DEFAULT_DATA = os.path.join(THIS_DIR, "dataset_split_631", "ordered", "split_1", "data.yaml")


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(THIS_DIR, p)


def _metrics(res) -> dict:
    b = res.box
    return {
        "precision": float(b.mp),
        "recall": float(b.mr),
        "mAP50": float(b.map50),
        "mAP50-95": float(b.map),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True, help="主训 best.pt（相对 finalmodel 或绝对路径）")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--project", required=True, help="精修结果输出目录（相对 finalmodel 或绝对路径）")
    ap.add_argument("--name", default="fold_1")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr0", type=float, default=1e-4)
    ap.add_argument("--lrf", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=20)
    # 主训一致的损失权重 / NMS IoU
    ap.add_argument("--box", type=float, default=7.5)
    ap.add_argument("--dfl", type=float, default=1.5)
    ap.add_argument("--cls", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.65)
    args = ap.parse_args()

    best = _abs(args.best)
    data = _abs(args.data)
    project = _abs(args.project)
    assert os.path.isfile(best), f"best.pt 不存在: {best}"
    assert os.path.isfile(data), f"data.yaml 不存在: {data}"

    print(f"[ft800] base weights = {best}")
    print(f"[ft800] data         = {data}")
    print(f"[ft800] project      = {project}")
    print(f"[ft800] {args.epochs}ep @imgsz{args.imgsz} batch={args.batch} "
          f"optimizer=AdamW lr0={args.lr0:g} lrf={args.lrf:g}（最终 lr≈{args.lr0 * args.lrf:g}）")

    model = YOLO(best)
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",        # 必须显式指定，否则 auto 会忽略 lr0
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=0.0,        # 续训精修无需 warmup，权重已是良好初始化
        cos_lr=True,
        # 关闭强增强：纯定位精修
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        multi_scale=0.0,
        close_mosaic=0,
        erasing=0.0,
        amp=True,
        patience=args.patience,
        box=args.box,
        dfl=args.dfl,
        cls=args.cls,
        iou=args.iou,
        project=project,
        name=args.name,
        seed=42,
        plots=True,
        exist_ok=True,
    )

    ft_dir = str(model.trainer.save_dir)
    ft_best = os.path.join(ft_dir, "weights", "best.pt")
    if not os.path.isfile(ft_best):
        ft_best = os.path.join(ft_dir, "weights", "last.pt")
    print(f"\n[ft800] 精修完成，最佳权重: {ft_best}")

    # ── 公平对比：分别在 640 / 800 评估 test ─────────────────────────────
    evaluator = YOLO(ft_best)
    results = {}
    for ev_imgsz in (640, 800):
        res = evaluator.val(
            data=data,
            split="test",
            imgsz=ev_imgsz,
            batch=args.batch,
            iou=args.iou,
            project=project,
            name=f"test_eval_{ev_imgsz}",
            plots=False,
            verbose=False,
            exist_ok=True,
        )
        results[f"test@{ev_imgsz}"] = _metrics(res)
        print(f"[ft800] test@{ev_imgsz}: {results[f'test@{ev_imgsz}']}")

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "yolov8s_p2_respam_sgf_ft800 (reuse best.pt, no full retrain)",
        "base_weights": best,
        "ft_weights": ft_best,
        "ft_params": {
            "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
            "optimizer": "AdamW", "lr0": args.lr0, "lrf": args.lrf,
            "warmup_epochs": 0.0, "cos_lr": True, "patience": args.patience,
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "multi_scale": 0.0,
            "close_mosaic": 0, "erasing": 0.0, "amp": True,
        },
        "loss_weights": {"box": args.box, "dfl": args.dfl, "cls": args.cls, "iou": args.iou},
        "test_metrics": results,
        "note": "test@640 与现有排名同口径用于公平对比；test@800 体现精修后最佳推理分辨率效果。",
    }
    out_json = os.path.join(project, "ft800_log.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"\n[ft800] 结果已写入: {out_json}")


if __name__ == "__main__":
    main()
