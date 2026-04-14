"""
=============================================================================
  3D PRINTING DEFECT DETECTION — PHASE 1
  Hardware Verification & Robust Data Pipeline
=============================================================================
  Target  : RTX 4070 Super (12 GB VRAM) | Windows | CUDA
  Dataset : 3D Printing Failure (spaghetti, stringing, zits)
  Classes : 3 (imbalanced — spaghetti dominates)
  Goal    : Verified DataLoader with WeightedRandomSampler + v2 augmentations
=============================================================================
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on Windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import v2


# =============================================================================
#  0.  GLOBAL CONFIGURATION
# =============================================================================

DATASET_ROOT  = "./dataset"  # must contain train/ val/ test/ subfolders
IMAGE_SIZE    = 224
BATCH_SIZE    = 128          # safe for 12 GB VRAM; try 192 if no OOM
NUM_WORKERS   = 0            # MUST be 0 on Windows to avoid BrokenPipe
PIN_MEMORY    = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# =============================================================================
#  1.  HARDWARE CHECK
# =============================================================================

def hardware_check() -> torch.device:
    print("=" * 60)
    print("  HARDWARE CHECK")
    print("=" * 60)
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  PyTorch     : {torch.__version__}")

    import torchvision
    print(f"  torchvision : {torchvision.__version__}")

    if not torch.cuda.is_available():
        print("\n  WARNING: CUDA NOT available — running on CPU.")
        print("     Verify your CUDA toolkit and driver versions.")
        return torch.device("cpu")

    device    = torch.device("cuda")
    gpu_name  = torch.cuda.get_device_name(0)
    props     = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory / (1024 ** 3)

    torch.cuda.empty_cache()
    free_mem  = (props.total_memory - torch.cuda.memory_reserved(0)) / (1024 ** 3)

    print(f"\n  CUDA available  (version {torch.version.cuda})")
    print(f"  GPU name    : {gpu_name}")
    print(f"  Total VRAM  : {total_mem:.2f} GB")
    print(f"  Free  VRAM  : {free_mem:.2f} GB")

    if "4070" not in gpu_name:
        print(f"\n  WARNING: Expected RTX 4070 Super but found: {gpu_name}")
    else:
        print(f"\n  RTX 4070 Super confirmed.")

    print("=" * 60)
    return device


# =============================================================================
#  2.  TRANSFORMS  (torchvision v2 API)
# =============================================================================

def build_transforms() -> dict:
    normalise = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),

        # Slight upscale then random crop for spatial variation
        v2.Resize(
            (IMAGE_SIZE + 16, IMAGE_SIZE + 16),
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        v2.RandomCrop(IMAGE_SIZE),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.3),
        v2.RandomRotation(degrees=30),

        # Photometric augmentations
        v2.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.05,
        ),
        v2.RandomGrayscale(p=0.1),

        # Random erasing — simulates partial occlusions and print debris
        v2.RandomErasing(
            p=0.3,
            scale=(0.02, 0.12),
            ratio=(0.3, 3.3),
            value=0,
        ),

        normalise,
    ])

    eval_tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        normalise,
    ])

    return {"train": train_tf, "val": eval_tf, "test": eval_tf}


# =============================================================================
#  3.  DATASET LOADING
# =============================================================================

def load_datasets(dataset_root: str, transforms: dict) -> dict:
    splits = {}
    for split in ("train", "val", "test"):
        path = os.path.join(dataset_root, split)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Expected directory not found: {path}\n"
                f"Please verify DATASET_ROOT = '{dataset_root}' is correct."
            )
        splits[split] = datasets.ImageFolder(
            root=path,
            transform=transforms[split],
        )

    class_names = splits["train"].classes
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Classes ({len(class_names)}) : {class_names}")
    for split, ds in splits.items():
        print(f"  {split:<6} samples : {len(ds)}")
    print("=" * 60)

    return splits


# =============================================================================
#  4.  WEIGHTED RANDOM SAMPLER
# =============================================================================

def build_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    targets      = np.array(dataset.targets)
    n_total      = len(targets)
    n_classes    = len(dataset.classes)
    class_counts = np.bincount(targets, minlength=n_classes).astype(float)

    print("\n  CLASS DISTRIBUTION (training set)")
    print("  " + "-" * 40)
    for idx, (name, cnt) in enumerate(zip(dataset.classes, class_counts.astype(int))):
        bar = "=" * int(30 * cnt / class_counts.max())
        print(f"  [{idx}] {name:<25} {cnt:>5}  {bar}")

    # Inverse-frequency weighting so every class is sampled equally
    class_weights  = n_total / (n_classes * class_counts)
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = n_total,
        replacement = True,
    )

    print("\n  Per-class sampling weights (higher = upsampled more):")
    for name, w in zip(dataset.classes, class_weights):
        print(f"    {name:<25} w = {w:.4f}")

    return sampler


# =============================================================================
#  5.  DATALOADER FACTORY
# =============================================================================

def build_dataloaders(datasets_dict: dict, batch_size: int = BATCH_SIZE) -> dict:
    train_sampler = build_weighted_sampler(datasets_dict["train"])

    loaders = {}

    loaders["train"] = DataLoader(
        datasets_dict["train"],
        batch_size  = batch_size,
        sampler     = train_sampler,   # replaces shuffle=True
        num_workers = NUM_WORKERS,     # 0 = Windows-safe
        pin_memory  = PIN_MEMORY,
        drop_last   = True,            # keeps batch size consistent
    )

    for split in ("val", "test"):
        loaders[split] = DataLoader(
            datasets_dict[split],
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = NUM_WORKERS,
            pin_memory  = PIN_MEMORY,
            drop_last   = False,
        )

    print("\n  DATALOADER CONFIGURATION")
    print(f"  Batch size   : {batch_size}")
    print(f"  num_workers  : {NUM_WORKERS}  (Windows-safe)")
    print(f"  pin_memory   : {PIN_MEMORY}")
    print(f"  Train batches: {len(loaders['train'])}")
    print(f"  Val   batches: {len(loaders['val'])}")
    print(f"  Test  batches: {len(loaders['test'])}")

    return loaders


# =============================================================================
#  6.  VISUALISATION HELPERS
# =============================================================================

def unnormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation and return HWC float numpy array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = tensor.cpu().clone() * std + mean
    img  = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def visualise_batch(
    loader:      DataLoader,
    class_names: list,
    n_show:      int = 16,
    save_path:   str = "phase1_batch_check.png",
) -> None:
    images, labels = next(iter(loader))
    n_show  = min(n_show, len(images))
    n_cols  = 8
    n_rows  = math.ceil(n_show / n_cols)

    # Count class distribution in this batch
    label_counts = {name: 0 for name in class_names}
    for lbl in labels[:n_show].tolist():
        label_counts[class_names[lbl]] += 1

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.0, n_rows * 2.0 + 1.5),
        facecolor="#1a1a2e",
    )
    fig.suptitle(
        "Phase 1 - Batch Visualisation\n"
        "(augmented, un-normalised, WeightedRandomSampler active)",
        color="white", fontsize=11, y=1.01,
    )

    # Flatten axes safely for both 1-row and multi-row grids
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    cmap  = plt.cm.get_cmap("tab10", len(class_names))
    c_map = {name: cmap(i) for i, name in enumerate(class_names)}

    for i in range(n_show):
        ax  = axes_flat[i]
        img = unnormalize(images[i])
        lbl = class_names[labels[i].item()]
        ax.imshow(img)
        ax.set_title(lbl, fontsize=7, color=c_map[lbl], pad=2, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(c_map[lbl])
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplot slots
    for j in range(n_show, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Legend showing sampled class distribution
    patches = [
        mpatches.Patch(color=c_map[n], label=f"{n}  (n={label_counts[n]})")
        for n in class_names
    ]
    fig.legend(
        handles        = patches,
        loc            = "lower center",
        ncol           = len(class_names),
        fontsize       = 9,
        facecolor      = "#1a1a2e",
        labelcolor     = "white",
        edgecolor      = "gray",
        bbox_to_anchor = (0.5, -0.04),
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=120, facecolor="#1a1a2e")
    plt.close(fig)

    print(f"\n  Batch visualisation saved -> '{save_path}'")
    print("\n  SAMPLED CLASS DISTRIBUTION (this batch)")
    print("  " + "-" * 40)
    for name, cnt in label_counts.items():
        bar = "=" * cnt
        print(f"  {name:<25} {cnt:>3}  {bar}")


# =============================================================================
#  7.  MAIN
# =============================================================================

def main():

    # 1. Hardware check
    device = hardware_check()

    # 2. Build transforms
    print("\n  Building transforms (torchvision v2) ...")
    transforms = build_transforms()
    print("  Transforms ready")

    # 3. Load datasets
    print(f"\n  Loading ImageFolder datasets from: {DATASET_ROOT}")
    try:
        splits = load_datasets(DATASET_ROOT, transforms)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    # 4. Build dataloaders
    print("\n  Building DataLoaders ...")
    loaders = build_dataloaders(splits)

    # 5. Smoke test
    print("\n  Running single-batch smoke test ...")
    images, labels = next(iter(loaders["train"]))

    assert images.shape == (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE), \
        f"Unexpected batch shape: {images.shape}"
    assert images.dtype == torch.float32, \
        f"Expected float32, got {images.dtype}"
    assert labels.max().item() < len(splits["train"].classes), \
        "Label index out of range"

    print(f"  Batch shape  : {tuple(images.shape)}")
    print(f"  dtype        : {images.dtype}")
    print(f"  Label range  : 0 - {labels.max().item()}")
    print(f"  Pixel range  : [{images.min():.3f}, {images.max():.3f}]  (normalised)")

    # 6. GPU transfer test
    if device.type == "cuda":
        images_gpu = images.to(device, non_blocking=True)
        print(f"  GPU transfer : {tuple(images_gpu.shape)} on {images_gpu.device}")
        del images_gpu
        torch.cuda.empty_cache()

    # 7. Evaluation gate — visual batch check
    print("\n  Generating batch visualisation ...")
    visualise_batch(
        loader      = loaders["train"],
        class_names = splits["train"].classes,
        n_show      = 16,
        save_path   = "phase1_batch_check.png",
    )

    print("\n" + "=" * 60)
    print("  PHASE 1 COMPLETE - all checks passed")
    print("  Review 'phase1_batch_check.png' then confirm Phase 2.")
    print("=" * 60)

    return device, splits, loaders


if __name__ == "__main__":
    main()