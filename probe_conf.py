"""排查混淆变量：不装 patch，纯统一 conf 阈值扫描。

若统一低阈值(如0.05~0.18)就能让 #7 达到 ~0.978，则之前 guard 的提升来自"降阈值"
而非 head-specific 过滤 —— head-specific 无独立贡献，结论需修正。
"""
import os
from ultralytics import YOLO
import infer_videos_diag as iv
import eval_p2guard as ep

CONFS = [0.05, 0.10, 0.15, 0.18, 0.22, 0.25, 0.28]
MAX_KEEP = 800


def main():
    all_vids = sorted(f for f in os.listdir(iv.VIDEO_DIR)
                      if os.path.splitext(f)[1].lower() in iv.VIDEO_EXTS)
    probes = {7: all_vids[6], 1: all_vids[0], 11: all_vids[10]}

    model = YOLO(ep.RESPAM)  # 全新实例，无 patch
    print("纯统一阈值（无 head-specific patch）")
    print(f"{'conf':6s} | {'#7 rate/big':14s} | {'#1 rate':8s} | {'#11 FP':8s}")
    print("-" * 60)
    for c in CONFS:
        r = {k: ep.run(model, os.path.join(iv.VIDEO_DIR, v), c, MAX_KEEP) for k, v in probes.items()}
        print(f"{c:<6.2f} | {r[7]['det_rate']:.3f}/{r[7]['big_boxes']:<7d} | "
              f"{r[1]['det_rate']:.3f}   | {r[11]['det_rate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
