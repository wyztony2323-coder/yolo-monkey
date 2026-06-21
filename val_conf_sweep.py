"""验证集(有GT)上量化 conf 工作点：一次 val 取全 conf 的 P/R/F1 曲线。

回答：降低 conf 抢回的召回，要以多少 precision 为代价？最佳部署 conf 在哪？
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO
import infer_videos_diag  # noqa: F401  注册自定义模块

DATA = r"D:\college\college3\monkey\yolo\finalmodel\dataset_split_631\ordered\split_1\data.yaml"
RESPAM = r"D:\college\college3\monkey\yolo\finalmodel\result_p2_respam_p2only_ft800\fold_1\weights\best.pt"
OUT = r"D:\college\college3\monkey\yolo\finalmodel\result_p2_respam_p2only_ft800\conf_sweep"
QUERY = [0.05, 0.10, 0.12, 0.15, 0.18, 0.22, 0.25, 0.30]


def main():
    os.makedirs(OUT, exist_ok=True)
    model = YOLO(RESPAM)
    m = model.val(data=DATA, imgsz=800, iou=0.65, conf=0.001, split="val",
                  plots=False, verbose=False, project=OUT, name="val", exist_ok=True)

    box = m.box
    f1 = np.asarray(box.f1_curve)
    p = np.asarray(box.p_curve)
    r = np.asarray(box.r_curve)
    if f1.ndim == 2:  # [nc, N] -> 单类取0；多类取均值
        f1, p, r = f1[0], p[0], r[0]
    n = len(f1)
    x = np.linspace(0, 1, n)

    print(f"mAP50={box.map50:.4f}  mAP50-95={box.map:.4f}\n")
    print(f"{'conf':6s} | {'P':7s} | {'R':7s} | {'F1':7s}")
    print("-" * 36)
    for c in QUERY:
        i = int(np.clip(round(c * (n - 1)), 0, n - 1))
        print(f"{c:<6.2f} | {p[i]:.4f} | {r[i]:.4f} | {f1[i]:.4f}")
    bi = int(np.argmax(f1))
    print("-" * 36)
    print(f"最佳F1 @ conf={x[bi]:.3f} : P={p[bi]:.4f} R={r[bi]:.4f} F1={f1[bi]:.4f}")

    plt.figure(figsize=(9, 5))
    plt.plot(x, p, label="Precision", color="#2980b9")
    plt.plot(x, r, label="Recall", color="#27ae60")
    plt.plot(x, f1, label="F1", color="#c0392b")
    plt.axvline(x[bi], ls="--", color="gray", label=f"best F1 conf={x[bi]:.3f}")
    plt.xlabel("confidence threshold")
    plt.ylabel("score")
    plt.title("respam_p2only_ft800  P/R/F1 vs conf (val)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    save = os.path.join(OUT, "pr_f1_vs_conf.png")
    plt.savefig(save, dpi=130, bbox_inches="tight")
    print(f"\n[OK] 曲线: {save}")


if __name__ == "__main__":
    main()
