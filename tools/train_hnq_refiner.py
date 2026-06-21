"""Train crop-based HNQ refiner from candidates.csv."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

FINALMODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINALMODEL))

from hnq_refiner import HNQDataset, HNQRefiner, hnq_loss


def make_balanced_sampler(dataset: HNQDataset) -> WeightedRandomSampler | None:
    labels = [int(r["label"]) for r in dataset.rows]
    if not labels:
        return None
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    weights = torch.tensor([0.5 / pos if y else 0.5 / neg for y in labels], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def evaluate(model, loader, device):
    model.eval()
    totals = {"loss": 0.0, "cls": 0.0, "quality": 0.0, "delta": 0.0}
    n = 0
    correct = 0
    total_cls = 0
    with torch.no_grad():
        for crop, geom, label, q, delta in loader:
            crop, geom, label, q, delta = crop.to(device), geom.to(device), label.to(device), q.to(device), delta.to(device)
            out = model(crop, geom)
            loss, parts = hnq_loss(out, label, q, delta)
            bs = crop.shape[0]
            totals["loss"] += float(loss) * bs
            for k in ("cls", "quality", "delta"):
                totals[k] += parts[k] * bs
            pred = (torch.sigmoid(out[0]) >= 0.5).float()
            correct += int((pred == label).sum().item())
            total_cls += int(label.numel())
            n += bs
    if n == 0:
        return {k: 0.0 for k in totals} | {"acc": 0.0}
    return {k: v / n for k, v in totals.items()} | {"acc": correct / max(1, total_cls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="HNQ candidates.csv")
    ap.add_argument("--out_dir", default=str(FINALMODEL / "result_hnq_refiner"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_balanced_sampler", action="store_true", help="disable label-balanced train sampling")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    train_ds = HNQDataset(args.csv, split="train", img_size=args.img_size)
    val_ds = HNQDataset(args.csv, split="val", img_size=args.img_size)
    sampler = None if args.no_balanced_sampler else make_balanced_sampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    print(f"[INFO] train={len(train_ds)} val={len(val_ds)} device={device} balanced_sampler={sampler is not None}")

    model = HNQRefiner().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_loss = float("inf")
    log_rows = []
    for ep in range(1, args.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "cls": 0.0, "quality": 0.0, "delta": 0.0}
        n = 0
        for crop, geom, label, q, delta in train_loader:
            crop, geom, label, q, delta = crop.to(device), geom.to(device), label.to(device), q.to(device), delta.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(crop, geom)
                loss, parts = hnq_loss(out, label, q, delta)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bs = crop.shape[0]
            sums["loss"] += float(loss.detach()) * bs
            for k in ("cls", "quality", "delta"):
                sums[k] += parts[k] * bs
            n += bs
        train_metrics = {f"train_{k}": v / max(1, n) for k, v in sums.items()}
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": ep, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        log_rows.append(row)
        print(
            f"{ep:03d}/{args.epochs} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.3f}",
            flush=True,
        )
        if row["val_loss"] < best_loss:
            best_loss = row["val_loss"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "val_loss": best_loss}, out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep}, out_dir / "last.pt")

    if log_rows:
        with open(out_dir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            wr.writeheader()
            wr.writerows(log_rows)
    (out_dir / "summary.json").write_text(
        json.dumps({"best_val_loss": best_loss, "train": len(train_ds), "val": len(val_ds)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] HNQ training done: {out_dir}")


if __name__ == "__main__":
    main()
