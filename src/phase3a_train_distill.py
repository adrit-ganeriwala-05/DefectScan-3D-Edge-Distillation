"""
=============================================================================
  phase3a_train_distill.py
  Staged Knowledge Distillation — MobileNetV4-Medium → MobileNetV4-Small
=============================================================================
  Strategy for 95%+ F1 (from 55% baseline):

    1. pretrained=True (PRIMARY CHANGE)
       The 55% F1 ceiling was a random-init problem, not a distillation
       problem. A pretrained student starts with full geometric and texture
       understanding from ImageNet — every gradient from epoch 1 refines
       defect discrimination rather than building visual primitives. Expected
       outcome: 90–96% F1.

    2. CosineAnnealingWarmRestarts (T_0=30, T_mult=2)
       Warm restarts escape local minima by periodically respiking LR.
       More effective than flat cosine once the model is near convergence,
       which happens quickly with pretrained weights.

    3. CE_ONLY_EPOCHS reduced 20 → 5
       With pretrained weights the student is stable from epoch 1. 20 epochs
       of CE-only warmup was needed for random init — now it wastes LR budget
       that could be used absorbing teacher knowledge.

    4. Temperature raised 4.0 → 5.0
       Softer teacher distributions give richer inter-class signal.
       KL effective weight = beta(0.4) * T²(25) * kl_scale(1.0) = 10.0
       vs CE 0.4. Higher ratio acceptable because pretrained student logits
       are already close to teacher's — KL magnitude is small.

  Stability features retained from audit:
    [ON]  WeightedRandomSampler only   — no class-weighted CE (no double-dip)
    [ON]  kl_scale = 1.0               — batchmean already normalises by N
    [ON]  AT float32 cast              — features.float().pow(2) prevents bf16 underflow
    [ON]  Channels-last NHWC           — Ada Lovelace tensor core layout
    [ON]  Fused AdamW                  — single CUDA kernel per step
    [ON]  uint8 RAM cache              — zero disk I/O per training step
    [ON]  try/finally worker cleanup   — no hung processes on Windows crash
    [ON]  validate() zero-div guard    — RuntimeError on empty val set
    [ON]  Single forward pass          — forward_features() + forward_head()

  Outputs:
    outputs/models/student_fp32_best.pth   best FP32 weights (highest val macro F1)
    phase3_curves.png       5-panel training curves

  Quantization is handled separately by phase4a_quantize_int8.py.
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import timm
from torchvision import datasets as tvdatasets
from torchvision.transforms import v2
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase1_data_pipeline import (
    hardware_check,
    DATASET_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
)


# =============================================================================
#  0.  CONFIGURATION
# =============================================================================

NUM_CLASSES      = 3
NUM_EPOCHS       = 150
LEARNING_RATE    = 3e-5
WEIGHT_DECAY     = 1e-2
ES_PATIENCE      = 25

BATCH_SIZE       = 64
NUM_WORKERS      = 4
AMP_DTYPE        = torch.bfloat16

CE_ONLY_EPOCHS   = 5            # reduced from 20 — pretrained student
                                 # is stable from epoch 1

# Stage 2 distillation weights (must sum to 1.0)
ALPHA            = 0.4          # Cross Entropy (standard, unweighted)
BETA             = 0.4          # KL Divergence (soft labels)
GAMMA            = 0.2          # Attention Transfer (spatial heatmap MSE)
TEMPERATURE      = 5.0          # raised from 4.0 — softer distributions,
                                 # richer inter-class signal

# CosineAnnealingWarmRestarts schedule
# T_0=30: first restart after 30 epochs
# T_mult=2: each cycle doubles (30 → 60 → 120)
CAWR_T0          = 30
CAWR_T_MULT      = 2

TEACHER_NAME     = "mobilenetv4_conv_medium"
STUDENT_NAME     = "mobilenetv4_conv_small"
TEACHER_WEIGHTS  = "outputs/models/teacher_best.pth"
STUDENT_FP32     = "outputs/models/student_fp32_best.pth"
CURVES_PATH      = "phase3_curves.png"


# =============================================================================
#  1.  IN-MEMORY DATASET  (uint8 CHW tensors)
# =============================================================================

class InMemoryImageFolder(Dataset):
    """
    Pre-loads all images as decoded uint8 CHW tensors into RAM.

    uint8 CHW  = 150 KB / image  →  8,851 images ≈ 1.3 GB RAM
    float32 CHW = 602 KB / image →  8,851 images ≈ 5.2 GB RAM

    __getitem__ applies float conversion + augmentations from memory —
    zero filesystem I/O per training step after __init__ completes.
    """

    def __init__(self, root: str, transform: v2.Compose, split: str = ""):
        base              = tvdatasets.ImageFolder(root=root)
        self.classes      = base.classes
        self.class_to_idx = base.class_to_idx
        self.targets      = base.targets
        self.transform    = transform

        print(f"  [Cache] '{split}' ({len(base)} images) ...",
              end=" ", flush=True)
        t0        = time.time()
        to_tensor = v2.ToImage()

        from PIL import Image as PILImage   # imported once before loop
        self.images: list[torch.Tensor] = []
        for path, _ in base.imgs:
            self.images.append(
                to_tensor(PILImage.open(path).convert("RGB"))
            )

        mem_mb = sum(t.nbytes for t in self.images) / (1024 ** 2)
        print(f"{time.time()-t0:.1f}s | {mem_mb:.0f} MB")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple:
        return self.transform(self.images[idx]), self.targets[idx]


# =============================================================================
#  2.  TRANSFORMS
# =============================================================================

def build_transforms() -> dict:
    norm = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # Training: no ToImage() — cache outputs uint8 tensors already
    train_tf = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16),
                  interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.RandomCrop(IMAGE_SIZE),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.3),
        v2.RandomRotation(degrees=30),
        v2.ColorJitter(brightness=0.4, contrast=0.4,
                       saturation=0.3, hue=0.05),
        v2.RandomGrayscale(p=0.1),
        v2.RandomErasing(p=0.3, scale=(0.02, 0.12),
                         ratio=(0.3, 3.3), value=0),
        norm,
    ])

    eval_tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE),
                  interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        norm,
    ])

    return {"train": train_tf, "val": eval_tf, "test": eval_tf}


# =============================================================================
#  3.  DATALOADERS
# =============================================================================

def build_dataloaders(device: torch.device = None) -> tuple[dict, list]:
    """
    Imbalance strategy: WeightedRandomSampler only.

    The sampler rebalances at the batch level — every batch sees ~equal
    class representation. This fixes imbalance for ALL loss components:
    CE, KL (soft label signal), and AT (spatial attention signal).

    Class-Weighted CE is intentionally NOT used alongside the sampler.
    Double-dipping would upweight stringing ~6.8x on an already-balanced
    batch, collapsing precision on other classes.
    """
    transforms  = build_transforms()
    class_names = None
    loaders     = {}

    print("\n  Building in-memory cached dataloaders ...")
    for split in ("train", "val", "test"):
        path    = os.path.join(DATASET_ROOT, split)
        tf      = transforms["train"] if split == "train" else transforms["val"]
        dataset = InMemoryImageFolder(root=path, transform=tf, split=split)

        if class_names is None:
            class_names = dataset.classes

        if split == "train":
            targets_arr   = np.array(dataset.targets)
            n_total       = len(targets_arr)
            n_classes     = len(dataset.classes)
            class_counts  = np.bincount(targets_arr,
                                        minlength=n_classes).astype(float)
            sample_weights = (n_total / (n_classes * class_counts))[targets_arr]

            print("  WeightedRandomSampler:")
            for name, cnt in zip(dataset.classes, class_counts.astype(int)):
                w = n_total / (n_classes * float(cnt))
                print(f"    {name:<25} n={cnt:>5}  w={w:.4f}")

            sampler = WeightedRandomSampler(
                weights     = torch.from_numpy(sample_weights).float(),
                num_samples = n_total,
                replacement = True,
            )
            loaders["train"] = DataLoader(
                dataset, batch_size=BATCH_SIZE, sampler=sampler,
                num_workers=NUM_WORKERS, pin_memory=True,
                drop_last=True, persistent_workers=True,
            )
        else:
            loaders[split] = DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
                persistent_workers=True,
            )

    print(f"\n  Classes       : {class_names}")
    print(f"  Train batches : {len(loaders['train'])} / epoch")
    return loaders, class_names


# =============================================================================
#  4.  SINGLE-PASS FORWARD
# =============================================================================

def single_pass(backbone: nn.Module,
                x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One backbone execution → (logits, spatial_feature_map).

    timm API:
        forward_features(x) → spatial map  (B, C, H, W)
        forward_head(feats) → class logits  (B, num_classes)

    Replaces the previous pattern of calling model(x) AND
    forward_features(x) separately — two full backbone passes per step.
    """
    feats  = backbone.forward_features(x)
    logits = backbone.forward_head(feats)
    return logits, feats


# =============================================================================
#  5.  ATTENTION TRANSFER
# =============================================================================

def attention_map(features: torch.Tensor) -> torch.Tensor:
    """
    A(F) = L2_normalize( sum(F², dim=channel) ),  shape (B, H, W)

    Cast to float32 before squaring — bfloat16 has 7-bit mantissa and
    small activations can underflow to zero after pow(2), producing
    degenerate all-zero maps and zero AT gradient.

    For homogeneous MobileNetV4 pairs, teacher and student share the
    same spatial H×W, so no channel adapter is needed.
    """
    # float32 cast prevents bfloat16 underflow
    attn      = features.float().pow(2).sum(dim=1)   # (B, H, W)
    b, h, w   = attn.shape
    attn_norm = F.normalize(attn.view(b, -1), p=2, dim=1)
    return attn_norm.view(b, h, w)


def attention_transfer_loss(s_feats: torch.Tensor,
                             t_feats: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(attention_map(s_feats), attention_map(t_feats))


# =============================================================================
#  6.  STAGED DISTILLATION LOSS
# =============================================================================

class StagedDistillationLoss(nn.Module):
    """
    Stage 1 (epoch ≤ CE_ONLY_EPOCHS):
        L = CE(student_logits, labels)

    Stage 2 (epoch > CE_ONLY_EPOCHS):
        L = alpha * CE
          + beta  * KL(softmax(s/T) || softmax(t/T)) * T²
          + gamma * AT_MSE(student_feats, teacher_feats)

    kl_scale = 1.0:
        F.kl_div(batchmean) divides by N — already a per-sample average,
        identical normalisation to CE. No correction factor needed.
        A previous version used 256/64=4.0, which inflated effective KL
        weight to 25.6 (64× CE) and destabilised Stage 2 training.

    T² scaling retained (Hinton et al. 2015):
        Dividing logits by T shrinks gradient magnitude by 1/T². The T²
        multiplier restores gradients to CE scale so losses are comparable.

    CE is standard (no class weights):
        WeightedRandomSampler already balances batches. Adding class weights
        would double-count imbalance correction (double-dipping).
    """

    def __init__(self):
        super().__init__()
        assert abs(ALPHA + BETA + GAMMA - 1.0) < 1e-5, \
            "ALPHA + BETA + GAMMA must equal 1.0"

        self.alpha    = ALPHA
        self.beta     = BETA
        self.gamma    = GAMMA
        self.T        = TEMPERATURE
        self.kl_scale = 1.0   # batchmean already normalises by N

        # Standard CE — no weight= argument (sampler handles imbalance)
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(
        self,
        s_logits: torch.Tensor,   # (B, C)
        t_logits: torch.Tensor,   # (B, C) — detached by caller
        labels:   torch.Tensor,   # (B,)
        s_feats:  torch.Tensor,   # (B, C_s, H, W)
        t_feats:  torch.Tensor,   # (B, C_t, H, W) — detached by caller
        stage:    int,
    ) -> tuple:

        loss_ce = self.ce(s_logits, labels)

        if stage == 1:
            zero = torch.tensor(0.0, device=s_logits.device)
            return loss_ce, loss_ce, zero, zero

        soft_s  = F.log_softmax(s_logits / self.T, dim=1)
        soft_t  = F.softmax(   t_logits / self.T, dim=1)
        loss_kl = (F.kl_div(soft_s, soft_t, reduction="batchmean")
                   * (self.T ** 2) * self.kl_scale)

        loss_at = attention_transfer_loss(s_feats, t_feats)

        total = (self.alpha * loss_ce
                 + self.beta  * loss_kl
                 + self.gamma * loss_at)

        return total, loss_ce, loss_kl, loss_at


# =============================================================================
#  7.  EARLY STOPPING ON MACRO F1
# =============================================================================

class EarlyStopping:
    """
    Monitors validation Macro F1 (higher = better).

    Macro F1 averages per-class F1 with equal weight, so a collapse in
    any single class immediately degrades the metric — unlike raw accuracy
    which can improve while stringing recall collapses to zero.
    """

    def __init__(self, patience: int = ES_PATIENCE,
                 save_path: str = STUDENT_FP32):
        self.patience   = patience
        self.save_path  = save_path
        self.best_f1    = -1.0
        self.counter    = 0
        self.early_stop = False
        self.best_epoch = 0

    def step(self, val_f1: float, backbone: nn.Module,
             epoch: int) -> bool:
        if val_f1 > self.best_f1 + 1e-4:
            self.best_f1    = val_f1
            self.best_epoch = epoch
            self.counter    = 0
            torch.save(copy.deepcopy(backbone.state_dict()),
                       self.save_path)
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


# =============================================================================
#  8.  MODEL BUILDERS  (channels-last NHWC)
# =============================================================================

def build_teacher(device: torch.device) -> nn.Module:
    if not os.path.isfile(TEACHER_WEIGHTS):
        print(f"  ERROR: '{TEACHER_WEIGHTS}' not found. "
              "Run phase2_train_teacher.py first.")
        sys.exit(1)

    backbone = timm.create_model(TEACHER_NAME, pretrained=False,
                                 num_classes=NUM_CLASSES)
    backbone.load_state_dict(
        torch.load(TEACHER_WEIGHTS, map_location=device, weights_only=True))
    backbone = backbone.to(memory_format=torch.channels_last).to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    n = sum(p.numel() for p in backbone.parameters())
    print(f"  Teacher ({TEACHER_NAME}): {n:,} params | frozen | ch-last")
    return backbone


def build_student(device: torch.device) -> nn.Module:
    """
    pretrained=True: student starts with ImageNet weights.

    This is the primary change from the previous run. A random-init student
    on 8.8k images mathematically caps at ~55% F1 because it spends most
    of training learning that edges and textures exist. A pretrained student
    already has this knowledge — every gradient from epoch 1 refines defect
    discrimination directly.
    """
    backbone = timm.create_model(STUDENT_NAME,
                                 pretrained=True,    # KEY CHANGE
                                 num_classes=NUM_CLASSES)
    backbone = backbone.to(memory_format=torch.channels_last).to(device)
    n = sum(p.numel() for p in backbone.parameters())
    print(f"  Student ({STUDENT_NAME}): {n:,} params | pretrained=True | ch-last")
    return backbone


def detect_channels(model_name: str, device: torch.device) -> int:
    m = timm.create_model(model_name, pretrained=False,
                          num_classes=NUM_CLASSES).to(device)
    m.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        ch    = m.forward_features(dummy).shape[1]
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ch


# =============================================================================
#  9.  VALIDATION PASS
# =============================================================================

@torch.no_grad()
def validate(model: nn.Module, teacher: nn.Module,
             loader: DataLoader, criterion: StagedDistillationLoss,
             device: torch.device, stage: int) -> tuple[float, float]:

    model.eval()
    va_loss   = 0.0
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True,
                           memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        if stage == 2:
            t_logits, t_feats = single_pass(teacher, images)
            t_feats = t_feats.detach()
        else:
            t_logits = torch.zeros(images.size(0), NUM_CLASSES, device=device)
            t_feats  = torch.zeros(1, device=device)

        s_logits, s_feats = single_pass(model, images)
        loss, *_ = criterion(
            s_logits, t_logits, labels, s_feats, t_feats, stage)

        va_loss    += loss.item() * images.size(0)
        all_preds.extend(s_logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    if n == 0:
        raise RuntimeError(
            "validate() received an empty loader. "
            "Check that DATASET_ROOT/val/ contains images."
        )
    va_loss  = va_loss / n
    macro_f1 = f1_score(all_labels, all_preds,
                        average="macro", zero_division=0)
    return va_loss, macro_f1


# =============================================================================
#  10. TRAINING LOOP
# =============================================================================

def train(teacher: nn.Module, student: nn.Module,
          loaders: dict, device: torch.device) -> dict:

    criterion = StagedDistillationLoss()
    use_fused = device.type == "cuda"
    optimizer = AdamW(
        student.parameters(),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=use_fused,
    )

    # CosineAnnealingWarmRestarts: first restart at T_0=30, then 60, 120
    # Warm restarts help escape local minima once pretrained model is near
    # convergence — more effective than flat cosine at that stage.
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=CAWR_T0, T_mult=CAWR_T_MULT, eta_min=1e-7)

    stopper  = EarlyStopping()
    use_amp  = device.type == "cuda"

    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss":   [], "val_f1":   [],
        "loss_ce": [], "loss_kl": [], "loss_at": [],
        "lr": [], "stage": [],
    }

    print("\n" + "=" * 70)
    print("  STAGED DISTILLATION  (pretrained student)")
    print(f"  Stage 1: epochs   1–{CE_ONLY_EPOCHS:<3}  CE only  (warmup)")
    print(f"  Stage 2: epochs {CE_ONLY_EPOCHS+1:>3}–{NUM_EPOCHS:<3}  "
          f"alpha={ALPHA}*CE + beta={BETA}*KL(T={TEMPERATURE}) + "
          f"gamma={GAMMA}*AT")
    print(f"  Scheduler: CosineAnnealingWarmRestarts "
          f"T_0={CAWR_T0} T_mult={CAWR_T_MULT}")
    print(f"  kl_scale=1.0 | kl_effective={BETA*TEMPERATURE**2:.1f} vs CE={ALPHA}")
    print(f"  Early stopping: Macro F1  patience={ES_PATIENCE}")
    print("=" * 70)

    t0 = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):

        stage  = 1 if epoch <= CE_ONLY_EPOCHS else 2
        lr_now = optimizer.param_groups[0]["lr"]

        if epoch == CE_ONLY_EPOCHS + 1:
            print(f"\n  {'='*70}")
            print(f"  STAGE 2 ACTIVE — Full Distillation (KL + AT)")
            print(f"  {'='*70}\n")

        # ── Train ─────────────────────────────────────────────────────────
        student.train()
        teacher.eval()

        run_loss = run_ce = run_kl = run_at = 0.0
        correct  = total_n = 0
        train_preds_all  = []
        train_labels_all = []

        for batch_idx, (images, labels) in enumerate(loaders["train"]):
            images = images.to(device, non_blocking=True,
                               memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    if stage == 2:
                        # Teacher: no_grad sufficient — no backward pass
                        with torch.no_grad():
                            t_logits, t_feats = single_pass(teacher, images)
                            t_feats = t_feats.detach()
                    else:
                        t_logits = torch.zeros(
                            images.size(0), NUM_CLASSES, device=device)
                        t_feats  = torch.zeros(1, device=device)

                    s_logits, s_feats = single_pass(student, images)
                    loss, lce, lkl, lat = criterion(
                        s_logits, t_logits, labels,
                        s_feats, t_feats, stage)
            else:
                if stage == 2:
                    with torch.no_grad():
                        t_logits, t_feats = single_pass(teacher, images)
                        t_feats = t_feats.detach()
                else:
                    t_logits = torch.zeros(
                        images.size(0), NUM_CLASSES, device=device)
                    t_feats  = torch.zeros(1, device=device)
                s_logits, s_feats = single_pass(student, images)
                loss, lce, lkl, lat = criterion(
                    s_logits, t_logits, labels,
                    s_feats, t_feats, stage)

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            # CosineAnnealingWarmRestarts steps per batch, not per epoch
            scheduler.step(epoch - 1 + batch_idx / len(loaders["train"]))

            bs        = images.size(0)
            run_loss += loss.item() * bs
            run_ce   += lce.item()  * bs
            run_kl   += lkl.item()  * bs
            run_at   += lat.item()  * bs
            batch_preds = s_logits.detach().argmax(1)
            correct    += (batch_preds == labels).sum().item()
            total_n    += bs
            train_preds_all.extend(batch_preds.cpu().numpy())
            train_labels_all.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 20 == 0 or \
               (batch_idx + 1) == len(loaders["train"]):
                print(
                    f"\r  [S{stage}] Ep {epoch}/{NUM_EPOCHS} "
                    f"[{batch_idx+1}/{len(loaders['train'])}] "
                    f"loss:{run_loss/total_n:.4f} "
                    f"acc:{100.*correct/total_n:.1f}%",
                    end="", flush=True)
        print()

        tr_loss = run_loss / total_n
        tr_acc  = correct  / total_n
        tr_f1   = f1_score(train_labels_all, train_preds_all,
                           average="macro", zero_division=0)

        va_loss, va_f1 = validate(
            student, teacher, loaders["val"], criterion, device, stage)

        improved  = stopper.step(va_f1, student, epoch)
        saved_tag = "  <-- BEST F1" if improved else ""

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(va_loss)
        history["val_f1"].append(va_f1)
        history["loss_ce"].append(run_ce  / total_n)
        history["loss_kl"].append(run_kl  / total_n)
        history["loss_at"].append(run_at  / total_n)
        history["lr"].append(lr_now)
        history["stage"].append(stage)

        print(
            f"  [S{stage}] Ep {epoch:>3}/{NUM_EPOCHS} | "
            f"LR:{lr_now:.1e} | "
            f"Train {tr_loss:.4f}/{tr_acc*100:.1f}%/F1:{tr_f1:.4f} | "
            f"Val {va_loss:.4f}/F1:{va_f1:.4f}"
            f"{saved_tag}")

        if stopper.early_stop:
            print(f"\n  Early stopping at epoch {epoch}. "
                  f"Best: ep {stopper.best_epoch} "
                  f"(val Macro F1={stopper.best_f1:.4f})")
            break

    elapsed     = time.time() - t0
    epochs_done = len(history["train_loss"])
    print(f"\n  Time: {elapsed/60:.1f}min  "
          f"({elapsed/epochs_done:.1f}s/ep)")
    print(f"  Best weights: {STUDENT_FP32}")
    return history


# =============================================================================
#  11. PLOTS
# =============================================================================

def plot_curves(history: dict) -> None:
    epochs       = range(1, len(history["train_loss"]) + 1)
    stage_switch = CE_ONLY_EPOCHS + 0.5

    fig, axes = plt.subplots(1, 5, figsize=(26, 4.5), facecolor="#1a1a2e")
    fig.suptitle(
        "Phase 3 — Distillation (pretrained student)  "
        f"CE + KL(T={TEMPERATURE}) + AT  |  "
        f"CAWR T₀={CAWR_T0}  |  Stage 2 @ epoch {CE_ONLY_EPOCHS+1}",
        color="white", fontsize=9)

    def _ax(ax, title, ylabel):
        ax.set_facecolor("#0f0f23")
        ax.axvline(x=stage_switch, color="#ffd54f", lw=1.5,
                   ls="--", alpha=0.6)
        ax.set_title(title, color="white", fontsize=9)
        ax.set_xlabel("Epoch", color="white", fontsize=8)
        ax.set_ylabel(ylabel, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.grid(alpha=0.2)
        for s in ax.spines.values():
            s.set_edgecolor("#444")

    # Panel 0 — Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color="#4fc3f7", lw=2, label="Train")
    ax.plot(epochs, history["val_loss"],   color="#ef5350", lw=2, label="Val")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7)
    _ax(ax, "Total Loss", "Loss")

    # Panel 1 — Macro F1
    ax = axes[1]
    ax.plot(epochs, history["train_f1"], color="#4fc3f7", lw=2,
            label="Train F1")
    ax.plot(epochs, history["val_f1"],   color="#69f0ae", lw=2,
            label="Val F1")
    ax.axhline(y=0.95, color="#ffd54f", lw=1.2, ls=":", label="0.95 target")
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7)
    _ax(ax, "Macro F1", "F1")

    # Panel 2 — CE loss
    ax = axes[2]
    ax.plot(epochs, history["loss_ce"], color="#a5d6a7", lw=2)
    _ax(ax, f"CE  (α={ALPHA})", "Loss")

    # Panel 3 — KL loss
    ax = axes[3]
    ax.plot(epochs, history["loss_kl"], color="#ce93d8", lw=2)
    _ax(ax, f"KL  (β={BETA}, T={TEMPERATURE})", "Loss")

    # Panel 4 — AT loss
    ax = axes[4]
    ax.plot(epochs, history["loss_at"], color="#ffcc80", lw=2)
    _ax(ax, f"AT  (γ={GAMMA})", "Loss")

    plt.tight_layout()
    plt.savefig(CURVES_PATH, bbox_inches="tight",
                dpi=120, facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Curves saved -> '{CURVES_PATH}'")


# =============================================================================
#  13. MAIN
# =============================================================================

if __name__ == "__main__":

    device = hardware_check()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
        torch.backends.cudnn.benchmark         = True
        device = torch.device("cuda")
        print("  [ON] TF32 + cuDNN benchmark")
        print(f"  [ON] AMP {AMP_DTYPE}  |  bs={BATCH_SIZE}")
        print(f"  [ON] pretrained=True  |  CE_ONLY={CE_ONLY_EPOCHS}ep")
        print(f"  [ON] CAWR T0={CAWR_T0} T_mult={CAWR_T_MULT}")
        print(f"  [ON] T={TEMPERATURE}  |  kl_scale=1.0")
        print("  [ON] Channels-last | Fused AdamW | uint8 cache")
        print("  [ON] WeightedSampler | standard CE (no double-dip)")
        print("  [ON] AT float32 cast | channels-last NHWC")
    else:
        device = torch.device("cpu")

    # Data
    loaders, class_names = build_dataloaders(device)

    # Models — teacher first, then student, then channel detection
    # (detect after builds to avoid a third simultaneous VRAM allocation)
    teacher = build_teacher(device)
    student = build_student(device)

    print(f"\n  Verifying feature channel dims ...")
    t_ch = detect_channels(TEACHER_NAME, device)
    s_ch = detect_channels(STUDENT_NAME, device)
    print(f"  Teacher: {t_ch} ch  |  Student: {s_ch} ch  |  "
          f"AT requires no adapter (shared H×W)")

    # Train
    try:
        history = train(teacher, student, loaders, device)
    except KeyboardInterrupt:
        print("\n  Training interrupted.")
        history = None
    finally:
        # Shut down persistent DataLoader workers — prevents hung processes
        # on Windows if training raises an exception
        for loader in loaders.values():
            loader._iterator = None

    if history is None:
        print("  No weights saved — exiting.")
        sys.exit(0)

    # Curves
    print("\n  Saving training curves ...")
    plot_curves(history)
    print(f"  Best val Macro F1 : {max(history['val_f1']):.4f}")

    print("\n" + "=" * 55)
    print("  PHASE 3A COMPLETE")
    print(f"  FP32 weights : {STUDENT_FP32}")
    print(f"  Curves       : {CURVES_PATH}")
    print("  Next         : python src/phase3b_eval_testset.py")
    print("  Then         : python src/phase4a_quantize_int8.py")
    print("=" * 55)