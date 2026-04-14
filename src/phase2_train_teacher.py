"""
=============================================================================
  3D PRINTING DEFECT DETECTION — PHASE 2
  Teacher Training (MobileNetV4-Conv-Medium) — Optimized for RTX 4070 Super
=============================================================================
  Hardware : RTX 4070 Super (Ada Lovelace, 12 GB VRAM)
  Model    : mobilenetv4_conv_medium (timm, pretrained=True)
  Dataset  : 3D Printing Failure — spaghetti / stringing / zits  (3 classes)

  Performance optimizations applied:
    [ON]  AMP bfloat16     — torch.autocast, native Ada Lovelace tensor cores
    [ON]  TF32             — matmul.allow_tf32, cudnn.allow_tf32
    [ON]  NUM_WORKERS = 4  — safe under if __name__ == "__main__" guard
    [ON]  BATCH_SIZE = 256 — fits in 12 GB with bfloat16 activations
    [OFF] torch.compile    — no Triton on Windows; would silently degrade
=============================================================================
"""

import os
import sys
import copy
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import timm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Phase 1 imports ──────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase1_data_pipeline import (
    build_transforms,
    load_datasets,
    build_dataloaders,
    hardware_check,
    DATASET_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# =============================================================================
#  0.  GLOBAL CONFIGURATION
# =============================================================================

NUM_CLASSES    = 3
NUM_EPOCHS     = 50
LEARNING_RATE  = 3e-4
WEIGHT_DECAY   = 1e-2
ES_PATIENCE    = 7
T_MAX          = 50        # CosineAnnealingLR full period

# ── Optimized settings ───────────────────────────────────────────────────────
BATCH_SIZE   = 256         # AMP halves activation memory — fits 12 GB
NUM_WORKERS  = 4           # Safe under __main__ guard on Windows
AMP_DTYPE    = torch.bfloat16   # Ada Lovelace has native bf16 tensor cores
                                 # bf16 does NOT need GradScaler (no underflow)

MODEL_NAME          = "mobilenetv4_conv_medium"
TEACHER_SAVE_PATH   = "outputs/models/teacher_best.pth"
CURVES_SAVE_PATH    = "phase2_training_curves.png"
CONFUSION_SAVE_PATH = "phase2_confusion_matrix.png"


# =============================================================================
#  1.  HARDWARE SETUP  (TF32 + AMP flags)
# =============================================================================

def configure_hardware() -> torch.device:
    """
    Run Phase 1 hardware check, then apply Ada Lovelace–specific settings:
      - TF32 for matmuls and cuDNN convolutions (free ~10-15% speedup)
      - cuDNN benchmark mode: auto-selects fastest conv algorithm per shape
    """
    device = hardware_check()

    if device.type == "cuda":
        # TF32 — uses tensor cores for float32 matmuls with 10-bit mantissa.
        # Full exponent range is preserved so numerical behaviour is stable.
        torch.backends.cuda.matmul.allow_tf32  = True
        torch.backends.cudnn.allow_tf32        = True

        # benchmark=True profiles cuDNN kernels on the first batch and caches
        # the fastest one. Since our batch size is fixed this is always a win.
        torch.backends.cudnn.benchmark         = True

        print("  [OPT] TF32 matmul        : ON")
        print("  [OPT] TF32 cuDNN         : ON")
        print("  [OPT] cuDNN benchmark    : ON")
        print(f"  [OPT] AMP dtype          : {AMP_DTYPE}")
        print(f"  [OPT] Batch size         : {BATCH_SIZE}")
        print(f"  [OPT] DataLoader workers : {NUM_WORKERS}")
        print("  [OFF] torch.compile      : skipped (no Triton on Windows)")

    return device


# =============================================================================
#  2.  EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Monitors validation loss. Saves best weights whenever val loss improves.
    Sets self.early_stop = True after `patience` epochs with no improvement.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4,
                 save_path: str = TEACHER_SAVE_PATH):
        self.patience   = patience
        self.min_delta  = min_delta
        self.save_path  = save_path
        self.best_loss  = float("inf")
        self.counter    = 0
        self.early_stop = False
        self.best_epoch = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Returns True if this is the new best epoch (weights saved)."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
            # Save to CPU to avoid holding an extra VRAM copy
            torch.save(copy.deepcopy(model.state_dict()), self.save_path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


# =============================================================================
#  3.  MODEL
# =============================================================================

def build_teacher(num_classes: int, device: torch.device) -> nn.Module:
    print(f"\n  Loading {MODEL_NAME} from timm (pretrained=True) ...")
    model = timm.create_model(
        MODEL_NAME,
        pretrained  = True,
        num_classes = num_classes,
    )
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")
    model = model.to(device)
    print(f"  Model on device  : {device}")
    return model


# =============================================================================
#  4.  TRAIN ONE EPOCH  (AMP bfloat16)
# =============================================================================

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
    total_epochs: int,
    use_amp:   bool,
) -> tuple:
    model.train()
    running_loss  = 0.0
    correct       = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        # ── AMP forward pass ─────────────────────────────────────────────────
        # torch.autocast casts eligible ops to bfloat16 automatically.
        # bfloat16 has the same exponent range as float32 so no GradScaler
        # is needed — the gradient scale is always 1.0 and never underflows.
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                logits = model(images)
                loss   = criterion(logits, labels)
        else:
            logits = model(images)
            loss   = criterion(logits, labels)

        loss.backward()

        # Gradient clipping — stabilises fine-tuning on small datasets
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss  += loss.item() * images.size(0)
        preds          = logits.detach().argmax(dim=1)
        correct       += (preds == labels).sum().item()
        total_samples += images.size(0)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(
                f"\r  Epoch [{epoch}/{total_epochs}] "
                f"Batch [{batch_idx+1}/{len(loader)}] "
                f"Loss: {running_loss/total_samples:.4f}  "
                f"Acc: {100.*correct/total_samples:.2f}%",
                end="", flush=True,
            )

    print()
    return running_loss / total_samples, correct / total_samples


# =============================================================================
#  5.  VALIDATE  (always float32 for accurate metric reporting)
# =============================================================================

@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple:
    """
    Validation always runs in full float32 — we want loss and accuracy
    numbers that are directly comparable across epochs without dtype noise.
    """
    model.eval()
    running_loss  = 0.0
    correct       = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)           # float32 — no autocast
        loss   = criterion(logits, labels)

        running_loss  += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        correct       += (preds == labels).sum().item()
        total_samples += images.size(0)

    return running_loss / total_samples, correct / total_samples


# =============================================================================
#  6.  FULL TRAINING LOOP
# =============================================================================

def train(
    model:        nn.Module,
    loaders:      dict,
    device:       torch.device,
    num_epochs:   int  = NUM_EPOCHS,
    use_amp:      bool = True,
) -> dict:

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-6)
    stopper   = EarlyStopping(patience=ES_PATIENCE, save_path=TEACHER_SAVE_PATH)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr":         [],
    }

    amp_label = f"bfloat16 autocast" if use_amp else "float32 (AMP off)"
    print("\n" + "=" * 60)
    print(f"  TEACHER TRAINING  |  AMP: {amp_label}")
    print("=" * 60)

    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]

        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer,
            device, epoch, num_epochs, use_amp,
        )
        va_loss, va_acc = validate(model, loaders["val"], criterion, device)
        scheduler.step()

        improved  = stopper.step(va_loss, model, epoch)
        saved_tag = "  <-- BEST SAVED" if improved else ""

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(lr_now)

        print(
            f"  Epoch {epoch:>3}/{num_epochs} | "
            f"LR: {lr_now:.2e} | "
            f"Train {tr_loss:.4f} / {tr_acc*100:.2f}% | "
            f"Val {va_loss:.4f} / {va_acc*100:.2f}%"
            f"{saved_tag}"
        )

        if stopper.early_stop:
            print(f"\n  Early stopping at epoch {epoch}.")
            print(f"  Best epoch : {stopper.best_epoch}  "
                  f"(val loss {stopper.best_loss:.4f})")
            break

    elapsed = time.time() - t0
    print(f"\n  Training time : {elapsed/60:.1f} min  "
          f"({elapsed/len(history['train_loss']):.1f}s / epoch)")
    print(f"  Weights saved : {TEACHER_SAVE_PATH}")

    return history


# =============================================================================
#  7.  EVALUATION GATE — CURVES + REPORT + CONFUSION MATRIX
# =============================================================================

def plot_training_curves(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1a1a2e")
    fig.suptitle(
        "Phase 2 — Teacher Training Curves (MobileNetV4-Medium + AMP bf16)",
        color="white", fontsize=12,
    )

    styles = dict(
        train=dict(color="#4fc3f7", lw=2, label="Train"),
        val  =dict(color="#ef5350", lw=2, label="Val"),
        bg   =dict(facecolor="#0f0f23"),
        grid =dict(alpha=0.2),
        spine=dict(edgecolor="#444"),
    )

    for ax, key, ylabel, title in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc")],
        ["Loss", "Accuracy (%)"],
        ["Loss", "Accuracy"],
    ):
        ax.set_facecolor("#0f0f23")
        tr_vals = history[key[0]]
        va_vals = history[key[1]]
        if "acc" in key[0]:
            tr_vals = [v * 100 for v in tr_vals]
            va_vals = [v * 100 for v in va_vals]
            ax.axhline(y=92, color="#ffd54f", lw=1.5,
                       ls="--", label="92% target")
        ax.plot(epochs, tr_vals, **styles["train"])
        ax.plot(epochs, va_vals, **styles["val"])
        ax.set_title(title, color="white")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1a1a2e", labelcolor="white")
        ax.grid(**styles["grid"])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(CURVES_SAVE_PATH, bbox_inches="tight", dpi=120, facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Training curves saved -> '{CURVES_SAVE_PATH}'")


@torch.no_grad()
def evaluate_on_test(
    model:        nn.Module,
    loader:       torch.utils.data.DataLoader,
    class_names:  list,
    device:       torch.device,
):
    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION (loading best saved weights)")
    print("=" * 60)

    state = torch.load(TEACHER_SAVE_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []

    for images, labels in loader:
        images  = images.to(device, non_blocking=True)
        logits  = model(images)
        preds   = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100
    print(f"\n  Test Accuracy : {accuracy:.2f}%")

    if accuracy >= 92.0:
        print("  GATE PASSED : >= 92% accuracy on test set.")
    else:
        print("  GATE NOT MET: below 92% — see tuning tips below.")

    # Per-class report
    print("\n  Classification Report:")
    print("  " + "-" * 52)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
    )
    for line in report.splitlines():
        print(f"  {line}")

    # Confusion matrix
    cm  = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#0f0f23")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_title("Confusion Matrix — Test Set (Teacher)", color="white",
                 fontsize=12, pad=12)
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("True",      color="white")
    ax.tick_params(colors="white")
    plt.setp(ax.get_xticklabels(), color="white", rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), color="white", rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_SAVE_PATH, bbox_inches="tight",
                dpi=120, facecolor="#1a1a2e")
    plt.close(fig)
    print(f"\n  Confusion matrix saved -> '{CONFUSION_SAVE_PATH}'")

    return accuracy


# =============================================================================
#  8.  TUNING TIPS  (printed if accuracy < 92%)
# =============================================================================

def print_tuning_tips():
    tips = """
  Tuning tips if accuracy < 92%:
  -------------------------------------------------------
  1. Lower LR        : Try LEARNING_RATE = 1e-4
  2. More epochs     : Set NUM_EPOCHS = 75, ES_PATIENCE = 10
  3. Reduce smoothing: Set label_smoothing = 0.05
  4. Check imbalance : Run Phase 1 visualisation again and
                       confirm sampler is balancing batches
  5. Unfreeze slowly : Consider freezing backbone for first
                       5 epochs then unfreezing all layers
"""
    print(tips)


# =============================================================================
#  9.  MAIN  (protected by __main__ — required for NUM_WORKERS > 0 on Windows)
# =============================================================================

if __name__ == "__main__":

    # 1. Hardware + optimisation flags
    device = configure_hardware()
    use_amp = device.type == "cuda"   # AMP only makes sense on GPU

    # 2. Data pipeline — reuse Phase 1, override batch size and workers
    print("\n  Building data pipeline ...")
    transforms = build_transforms()
    splits     = load_datasets(DATASET_ROOT, transforms)

    # Override Phase 1 defaults with optimised values for Phase 2
    loaders = build_dataloaders(
        splits,
        batch_size=BATCH_SIZE,    # 256 with AMP
    )
    # Rebuild train loader with correct num_workers (Phase 1 used 0)
    from torch.utils.data import DataLoader
    from phase1_data_pipeline import build_weighted_sampler

    train_sampler = build_weighted_sampler(splits["train"])
    loaders["train"] = DataLoader(
        splits["train"],
        batch_size  = BATCH_SIZE,
        sampler     = train_sampler,
        num_workers = NUM_WORKERS,    # 4 — safe under __main__ guard
        pin_memory  = True,
        drop_last   = True,
        persistent_workers = True,    # keep worker processes alive between
                                      # epochs — eliminates per-epoch
                                      # spawn overhead on Windows
    )

    class_names = splits["train"].classes
    print(f"  Classes : {class_names}")

    # 3. Model
    model = build_teacher(NUM_CLASSES, device)

    # 4. Train
    history = train(model, loaders, device, use_amp=use_amp)

    # 5. Evaluation gate
    print("\n  Generating training curves ...")
    plot_training_curves(history)

    best_val = max(history["val_acc"]) * 100
    print(f"\n  Best val accuracy during training : {best_val:.2f}%")

    test_acc = evaluate_on_test(model, loaders["test"], class_names, device)

    if test_acc < 92.0:
        print_tuning_tips()

    print("\n" + "=" * 60)
    print("  PHASE 2 COMPLETE")
    print(f"  Test accuracy    : {test_acc:.2f}%")
    print(f"  Best weights     : {TEACHER_SAVE_PATH}")
    print(f"  Training curves  : {CURVES_SAVE_PATH}")
    print(f"  Confusion matrix : {CONFUSION_SAVE_PATH}")
    print("=" * 60)