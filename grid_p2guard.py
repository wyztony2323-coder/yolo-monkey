"""P2-guard 参数网格扫描：扫 p2_conf × p2_max_area，固定 main_conf。

探针：
  #7 / #1 = 召回探针（有真实目标，检测率越高越好）
  #11「算法未识别到」= 假阳性探针（标注者都没识别到，检测率涨起来=误检）
最佳工作点 = #11 不涨的前提下 #7/#1 最高。
"""
from __future__ import annotations

import itertools
import os

from ultralytics import YOLO

import infer_videos_diag as iv
import eval_p2guard as ep

P2_CONFS = [0.12, 0.15, 0.18, 0.22, 0.28]
P2_AREAS = [0.05, 1.0]   # 0.05=限面积, 1.0=不限（看面积约束是否必要）
MAIN_CONF = 0.28
MAX_KEEP = 800


def main():
    all_vids = sorted(f for f in os.listdir(iv.VIDEO_DIR)
                      if os.path.splitext(f)[1].lower() in iv.VIDEO_EXTS)
    probes = {7: all_vids[6], 1: all_vids[0], 11: all_vids[10]}  # #7召回 #1召回 #11FP
    print("探针: #7/#1=召回(越高越好)  #11=假阳性(越低越好)\n")

    model = YOLO(ep.RESPAM)

    # baseline 参照（统一 conf=0.25）
    print(f"{'config':28s} | {'#7 rate/big':14s} | {'#1 rate':8s} | {'#11 FP':8s}")
    print("-" * 70)
    base = {k: ep.run(model, os.path.join(iv.VIDEO_DIR, v), 0.25, MAX_KEEP) for k, v in probes.items()}
    print(f"{'baseline conf=0.25':28s} | {base[7]['det_rate']:.3f}/{base[7]['big_boxes']:<7d} | "
          f"{base[1]['det_rate']:.3f}   | {base[11]['det_rate']:.3f}")
    print("-" * 70)

    rows = []
    for pc, pa in itertools.product(P2_CONFS, P2_AREAS):
        ep.install_p2guard(model, pc, MAIN_CONF, pa)
        r = {k: ep.run(model, os.path.join(iv.VIDEO_DIR, v), 0.05, MAX_KEEP) for k, v in probes.items()}
        cfg = f"p2={pc} main={MAIN_CONF} area={pa}"
        print(f"{cfg:28s} | {r[7]['det_rate']:.3f}/{r[7]['big_boxes']:<7d} | "
              f"{r[1]['det_rate']:.3f}   | {r[11]['det_rate']:.3f}", flush=True)
        rows.append((pc, pa, r))

    print("\n[OK] 网格完成。判读: 选 #11 接近 baseline(无明显上涨) 且 #7/#1 最高的组合。")


if __name__ == "__main__":
    main()
