"""Train an ImageNet-pretrained ResNet18 for the COMP9517 project.

Run this script from the project root, for example:

    python src/train_pretrained.py --run-name pretrained_freeze --freeze --epochs 20
    python src/train_pretrained.py --run-name pretrained_finetune --no-freeze --epochs 20

Outputs:
    results/models/{run_name}_best.pth
    results/models/{run_name}_last.pth
    results/logs/{run_name}_training_log.csv
    results/logs/{run_name}_summary.json
    results/figures/{run_name}_loss_curve.png
    results/figures/{run_name}_accuracy_curve.png
    results/figures/{run_name}_macro_f1_curve.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import make_loader
from pretrained_model import build_pretrained_resnet18


PROJECT_ROOT = Path.cwd()
NUM_CLASSES = 500
SEED = 9517


def check_project_root() -> None:
    required = [
        PROJECT_ROOT / "data" / "splits" / "train_paths.csv",
        PROJECT_ROOT / "data" / "splits" / "val_paths.csv",
        PROJECT_ROOT / "src" / "dataset.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run this script from the project root. Missing:\n  - "
            + "\n  - ".join(missing)
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_portable_csv(csv_path: str | Path) -> Path:
    """Create a temporary platform-compatible CSV only when necessary.

    The supplied split CSVs contain Windows-style backslashes. They work on
    Windows, but on Linux/Colab/macOS they need to be converted to the local
    path separator. The original fixed split file is never changed.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Split CSV is empty: {csv_path}")

    first_original = Path(str(df.iloc[0]["image_path"]))
    if first_original.exists():
        return csv_path

    converted = df.copy()
    converted["image_path"] = converted["image_path"].astype(str).map(
        lambda value: value.replace("\\", os.sep).replace("/", os.sep)
    )

    sample_paths = [Path(p) for p in converted["image_path"].head(20)]
    if not all(path.exists() for path in sample_paths):
        raise FileNotFoundError(
            "Images could not be found. Check that data/raw/train_mini and "
            "data/raw/val are present, that there is no extra nested folder, "
            "and that you are running from the project root.\n"
            f"First unresolved path: {sample_paths[0]}"
        )

    cache_dir = PROJECT_ROOT / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{csv_path.stem}_portable.csv"
    converted.to_csv(out_path, index=False)
    print(f"[path compatibility] Using temporary CSV: {out_path}")
    return out_path


def create_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    if enabled:
        try:
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        except AttributeError:
            return torch.cuda.amp.autocast()
    return nullcontext()


def batch_topk_counts(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    k = min(5, logits.shape[1])
    topk = logits.topk(k=k, dim=1).indices
    top1_correct = int((topk[:, 0] == targets).sum().item())
    top5_correct = int(topk.eq(targets.view(-1, 1)).any(dim=1).sum().item())
    return top1_correct, top5_correct


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    amp_enabled: bool,
    log_every: int,
) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    top1_count = 0
    top5_count = 0

    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            with autocast_context(amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # A disabled torch.amp.GradScaler is supposed to be a no-op
            # passthrough to backward()/optimizer.step(), but on this
            # environment's non-CUDA (MPS) backend it silently desyncs
            # BatchNorm running stats: training-time metrics look normal,
            # yet eval-mode accuracy afterwards collapses to near zero.
            # Skipping the scaler entirely when not using CUDA AMP avoids it.
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        top1, top5 = batch_topk_counts(logits.detach(), targets)
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        top1_count += top1
        top5_count += top5

        if log_every > 0 and batch_index % log_every == 0:
            print(
                f"    batch {batch_index:4d}/{len(loader)} | "
                f"loss={loss_sum / sample_count:.4f} | "
                f"top1={top1_count / sample_count:.4f}"
            )

    return {
        "train_loss": loss_sum / sample_count,
        "train_top1": top1_count / sample_count,
        "train_top5": top5_count / sample_count,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    top1_count = 0
    top5_count = 0
    all_true = []
    all_pred = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_context(amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = targets.size(0)
        top1, top5 = batch_topk_counts(logits, targets)
        predictions = logits.argmax(dim=1)

        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        top1_count += top1
        top5_count += top5
        all_true.extend(targets.cpu().tolist())
        all_pred.extend(predictions.cpu().tolist())

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        all_true,
        all_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "val_loss": loss_sum / sample_count,
        "val_top1": top1_count / sample_count,
        "val_top5": top5_count / sample_count,
        "val_macro_precision": float(precision),
        "val_macro_recall": float(recall),
        "val_macro_f1": float(macro_f1),
    }


def validate_via_subprocess(
    model: nn.Module,
    val_csv: str | Path,
    num_classes: int,
    image_size: int,
    batch_size: int,
    tmp_checkpoint_path: Path,
) -> Dict[str, float]:
    """Run validation in a fresh subprocess instead of in-process.

    Workaround for an environment-specific bug on this project's Apple
    Silicon / MPS development machine: calling a torch.no_grad() validation
    function in the SAME process that just ran train_one_epoch() silently
    corrupts predictions (near-random accuracy on correctly trained weights).
    Isolated repros confirmed the exact same checkpoint evaluates correctly
    when reloaded in a fresh process. See src/_mps_eval_worker.py.
    """
    tmp_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "num_classes": num_classes}, tmp_checkpoint_path)

    worker_path = Path(__file__).resolve().parent / "_mps_eval_worker.py"
    result = subprocess.run(
        [
            sys.executable,
            str(worker_path),
            "--checkpoint",
            str(tmp_checkpoint_path),
            "--val-csv",
            str(val_csv),
            "--num-classes",
            str(num_classes),
            "--image-size",
            str(image_size),
            "--batch-size",
            str(batch_size),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    last_line = result.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_macro_f1: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_macro_f1": best_val_macro_f1,
            "num_classes": args.num_classes,
            "run_name": args.run_name,
            "freeze_backbone": args.freeze,
            "image_size": args.image_size,
            "config": vars(args),
        },
        path,
    )


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_log_and_curves(records, run_name: str) -> None:
    logs_dir = PROJECT_ROOT / "results" / "logs"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    log_path = logs_dir / f"{run_name}_training_log.csv"
    df.to_csv(log_path, index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"{run_name}: loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"{run_name}_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_top1"], label="Train top-1")
    plt.plot(df["epoch"], df["val_top1"], label="Validation top-1")
    plt.plot(df["epoch"], df["val_top5"], label="Validation top-5")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{run_name}: accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"{run_name}_accuracy_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["val_macro_f1"], label="Validation macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(f"{run_name}: validation macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"{run_name}_macro_f1_curve.png", dpi=200)
    plt.close()

    print(f"Saved training log: {log_path}")
    print(f"Saved training curves under: {figures_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="pretrained_finetune")
    parser.add_argument("--train-csv", default="data/splits/train_paths.csv")
    parser.add_argument("--val-csv", default="data/splits/val_paths.csv")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument("--freeze", dest="freeze", action="store_true")
    freeze_group.add_argument("--no-freeze", dest="freeze", action="store_false")
    parser.set_defaults(freeze=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_project_root()
    set_seed(args.seed)

    # Freezing the backbone leaves only the new linear head to train, which
    # tolerates a higher learning rate. Fine-tuning the whole pretrained
    # network needs a smaller learning rate so it does not destroy the
    # pretrained features.
    if args.learning_rate is None:
        args.learning_rate = 1e-3 if args.freeze else 1e-4

    train_csv = make_portable_csv(args.train_csv)
    val_csv = make_portable_csv(args.val_csv)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    print(f"Device: {device}")
    print(f"Mixed precision: {amp_enabled}")
    print(f"Run name: {args.run_name}")
    print(f"Freeze backbone: {args.freeze}")
    print(f"Learning rate: {args.learning_rate}")

    train_loader = make_loader(
        csv_path=train_csv,
        split="train",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        augment=True,
        shuffle=True,
    )
    val_loader = make_loader(
        csv_path=val_csv,
        split="val",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        augment=False,
        shuffle=False,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    model = build_pretrained_resnet18(args.num_classes, freeze_backbone=args.freeze).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_trainable:,} / {num_total:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=1e-6,
    )
    scaler = create_grad_scaler(amp_enabled)

    models_dir = PROJECT_ROOT / "results" / "models"
    logs_dir = PROJECT_ROOT / "results" / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / f"{args.run_name}_best.pth"
    last_path = models_dir / f"{args.run_name}_last.pth"

    start_epoch = 1
    best_val_macro_f1 = -1.0
    best_epoch = 0
    no_improvement = 0
    records = []

    log_path = logs_dir / f"{args.run_name}_training_log.csv"
    if args.resume:
        checkpoint = safe_torch_load(Path(args.resume), device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_macro_f1 = float(checkpoint.get("best_val_macro_f1", -1.0))
        if log_path.exists():
            records = pd.read_csv(log_path).to_dict("records")
        print(f"Resumed from epoch {start_epoch - 1}: {args.resume}")

    total_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            log_every=args.log_every,
        )
        if device.type == "mps":
            val_metrics = validate_via_subprocess(
                model=model,
                val_csv=val_csv,
                num_classes=args.num_classes,
                image_size=args.image_size,
                batch_size=args.batch_size,
                tmp_checkpoint_path=models_dir / f"{args.run_name}_epoch_tmp.pth",
            )
        else:
            val_metrics = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
            )

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        epoch_time = time.time() - epoch_start

        record = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "learning_rate": current_lr,
            "epoch_time_sec": epoch_time,
        }
        records.append(record)
        save_log_and_curves(records, args.run_name)

        print(
            f"  train_loss={record['train_loss']:.4f} "
            f"train_top1={record['train_top1']:.4f} "
            f"val_loss={record['val_loss']:.4f} "
            f"val_top1={record['val_top1']:.4f} "
            f"val_top5={record['val_top5']:.4f} "
            f"val_macro_f1={record['val_macro_f1']:.4f} "
            f"time={epoch_time:.1f}s"
        )

        improved = record["val_macro_f1"] > best_val_macro_f1
        if improved:
            best_val_macro_f1 = record["val_macro_f1"]
            best_epoch = epoch
            no_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_macro_f1,
                args,
            )
            print(f"  New best checkpoint: {best_path}")
        else:
            no_improvement += 1

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_macro_f1,
            args,
        )

        if args.patience > 0 and no_improvement >= args.patience:
            print(
                f"Early stopping: validation macro-F1 did not improve for "
                f"{args.patience} epochs."
            )
            break

    total_time = time.time() - total_start
    log_df = pd.DataFrame(records)
    best_row = log_df.loc[log_df["val_macro_f1"].idxmax()].to_dict()

    summary = {
        "run_name": args.run_name,
        "pretrained": True,
        "torchvision_weights": "IMAGENET1K_V1",
        "freeze_backbone": args.freeze,
        "num_classes": args.num_classes,
        "num_train_images": len(train_loader.dataset),
        "num_val_images": len(val_loader.dataset),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "trainable_parameters": num_trainable,
        "total_parameters": num_total,
        "best_epoch": int(best_row["epoch"]),
        "best_val_top1": float(best_row["val_top1"]),
        "best_val_top5": float(best_row["val_top5"]),
        "best_val_macro_f1": float(best_row["val_macro_f1"]),
        "total_training_time_sec": total_time,
        "best_checkpoint": str(best_path).replace("\\", "/"),
        "last_checkpoint": str(last_path).replace("\\", "/"),
    }

    summary_path = logs_dir / f"{args.run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTraining complete.")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
