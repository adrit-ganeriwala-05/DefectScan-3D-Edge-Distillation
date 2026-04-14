"""
=============================================================================
  3D PRINTING DEFECT DETECTION — PHASE 2B
  Teacher Visual Sanity Check
=============================================================================
    Loads outputs/models/teacher_best.pth and runs 16 random test images through the model.
  Displays true label, predicted label, confidence, and soft label bar chart
  for each image in a 4x4 grid saved to teacher_visual_sanity_check.png
=============================================================================
"""

import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")  # must be before any other matplotlib import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import timm

from torchvision import datasets
from torchvision.transforms import v2

# ── Phase 1 imports ──────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase1_data_pipeline import DATASET_ROOT, IMAGENET_MEAN, IMAGENET_STD

# =============================================================================
#  CONFIGURATION
# =============================================================================

TEACHER_WEIGHTS  = "outputs/models/teacher_best.pth"
SAVE_PATH        = "teacher_visual_sanity_check.png"
NUM_IMAGES       = 16
IMAGE_SIZE       = 224
NUM_CLASSES      = 3
RANDOM_SEED      = 42
MODEL_NAME       = "mobilenetv4_conv_medium"

# =============================================================================
#  1.  CLEAN EVAL TRANSFORM  (no augmentation)
# =============================================================================

def build_eval_transform() -> v2.Compose:
    """
    Deterministic transform for visual evaluation.
    Resize to 224x224 + ImageNet normalisation only.
    No augmentations — we want to see exactly what the model sees.
    """
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# =============================================================================
#  2.  MODEL LOADER
# =============================================================================

def load_teacher(num_classes: int, weights_path: str,
                 device: torch.device) -> nn.Module:
    print(f"  Loading {MODEL_NAME} ...")
    model = timm.create_model(
        MODEL_NAME,
        pretrained  = False,      # weights come from our saved file
        num_classes = num_classes,
    )
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"  Weights loaded from : {weights_path}")
    print(f"  Model on device     : {device}")
    return model


# =============================================================================
#  3.  UN-NORMALISE HELPER
# =============================================================================

def unnormalize(tensor: torch.Tensor) -> np.ndarray:
    """CHW float tensor → HWC uint8 numpy array for imshow."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD ).view(3, 1, 1)
    img  = tensor.cpu().clone() * std + mean
    img  = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


# =============================================================================
#  4.  SAMPLE + INFER
# =============================================================================

@torch.no_grad()
def sample_and_infer(
    model:       nn.Module,
    dataset:     datasets.ImageFolder,
    device:      torch.device,
    n:           int = NUM_IMAGES,
    seed:        int = RANDOM_SEED,
) -> list:
    """
    Randomly select n images from the dataset, run them through the model,
    and return a list of result dicts containing everything needed to plot.
    """
    random.seed(seed)
    indices = random.sample(range(len(dataset)), n)

    results = []
    for idx in indices:
        tensor, true_label = dataset[idx]          # already transformed
        input_batch = tensor.unsqueeze(0).to(device)

        logits      = model(input_batch)            # (1, C)
        probs       = torch.softmax(logits, dim=1).squeeze(0).cpu()  # (C,)
        pred_label  = probs.argmax().item()
        confidence  = probs[pred_label].item() * 100

        results.append({
            "image"      : tensor,                  # CHW float, normalised
            "true_label" : true_label,
            "pred_label" : pred_label,
            "confidence" : confidence,
            "probs"      : probs.numpy(),           # (C,) for soft label bars
        })

    return results


# =============================================================================
#  5.  VISUALISATION
# =============================================================================

def plot_sanity_grid(
    results:     list,
    class_names: list,
    save_path:   str = SAVE_PATH,
):
    """
    4x4 grid.  Each cell contains:
      - The un-normalised RGB image
      - Title: True / Pred / Confidence  (green=correct, red=wrong)
      - A small soft-label probability bar chart below each image
    """
    n_cols = 4
    n_rows = 4
    assert len(results) == n_cols * n_rows, "Expected exactly 16 results"

    # Each image cell gets an image subplot + a thin bar chart subplot
    # We use GridSpec with height_ratios so bars don't crowd the images
    fig = plt.figure(figsize=(n_cols * 3.6, n_rows * 3.8), facecolor="#12121f")
    outer = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.55,
        wspace=0.25,
    )

    correct_count = 0

    for i, res in enumerate(results):
        row = i // n_cols
        col = i  % n_cols

        # Each cell: image on top, bar chart on bottom
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec   = outer[row, col],
            height_ratios  = [4, 1],
            hspace         = 0.08,
        )
        ax_img = fig.add_subplot(inner[0])
        ax_bar = fig.add_subplot(inner[1])

        # ── Image ─────────────────────────────────────────────────────────
        img = unnormalize(res["image"])
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        true_name = class_names[res["true_label"]]
        pred_name = class_names[res["pred_label"]]
        correct   = res["true_label"] == res["pred_label"]
        title_col = "#69f0ae" if correct else "#ff5252"   # green / red
        correct_count += int(correct)

        # Border colour mirrors title
        for spine in ax_img.spines.values():
            spine.set_edgecolor(title_col)
            spine.set_linewidth(2.5)

        ax_img.set_title(
            f"True : {true_name}\n"
            f"Pred : {pred_name}  ({res['confidence']:.1f}%)",
            fontsize  = 7.5,
            color     = title_col,
            fontweight= "bold",
            pad       = 4,
        )

        # ── Soft-label bar chart ───────────────────────────────────────────
        # Shows the full probability distribution — key for Phase 3 distillation
        bar_colors = []
        for j in range(len(class_names)):
            if j == res["true_label"]:
                bar_colors.append("#ffd54f")   # gold  = ground truth class
            elif j == res["pred_label"]:
                bar_colors.append(title_col)   # green/red = predicted class
            else:
                bar_colors.append("#546e7a")   # grey  = other classes

        bars = ax_bar.barh(
            range(len(class_names)),
            res["probs"] * 100,
            color  = bar_colors,
            height = 0.6,
        )
        ax_bar.set_xlim(0, 100)
        ax_bar.set_yticks(range(len(class_names)))
        ax_bar.set_yticklabels(
            [n[:4] for n in class_names],    # truncate to 4 chars for space
            fontsize=5.5, color="white",
        )
        ax_bar.set_xticks([])
        ax_bar.set_facecolor("#1a1a2e")
        for spine in ax_bar.spines.values():
            spine.set_visible(False)

        # Probability label on the longest bar
        for j, (bar, prob) in enumerate(zip(bars, res["probs"])):
            if prob > 0.08:
                ax_bar.text(
                    prob * 100 + 1, bar.get_y() + bar.get_height() / 2,
                    f"{prob*100:.1f}%",
                    va="center", fontsize=5, color="white",
                )

    # ── Summary title ──────────────────────────────────────────────────────
    fig.suptitle(
        f"Teacher Sanity Check — {MODEL_NAME}\n"
        f"{correct_count}/{len(results)} correct  "
        f"({correct_count/len(results)*100:.1f}% on this sample)  |  "
        f"Green border = correct   Red border = incorrect\n"
        f"Gold bar = ground truth class    Coloured bar = predicted class",
        color="white", fontsize=10, y=1.01,
    )

    plt.savefig(save_path, bbox_inches="tight", dpi=130, facecolor="#12121f")
    plt.close(fig)
    print(f"\n  Sanity check grid saved -> '{save_path}'")
    print(f"  Sample accuracy : {correct_count}/{len(results)} "
          f"({correct_count/len(results)*100:.1f}%)")


# =============================================================================
#  6.  MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  PHASE 2B — TEACHER VISUAL SANITY CHECK")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    # Verify weights file exists before doing anything else
    if not os.path.isfile(TEACHER_WEIGHTS):
        print(f"\n  ERROR: '{TEACHER_WEIGHTS}' not found.")
        print("  Run phase2_train_teacher.py first.")
        sys.exit(1)

    # Clean eval dataset — no augmentations
    test_path = os.path.join(DATASET_ROOT, "test")
    if not os.path.isdir(test_path):
        print(f"\n  ERROR: test split not found at '{test_path}'")
        sys.exit(1)

    transform   = build_eval_transform()
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    class_names  = test_dataset.classes
    print(f"  Test set    : {len(test_dataset)} images")
    print(f"  Classes     : {class_names}")

    # Load teacher
    model = load_teacher(NUM_CLASSES, TEACHER_WEIGHTS, device)

    # Sample 16 images and run inference
    print(f"\n  Sampling {NUM_IMAGES} random test images (seed={RANDOM_SEED}) ...")
    results = sample_and_infer(model, test_dataset, device)

    # Print quick summary to terminal
    print("\n  Per-image results:")
    print("  " + "-" * 52)
    for i, r in enumerate(results):
        true_name = class_names[r["true_label"]]
        pred_name = class_names[r["pred_label"]]
        status    = "OK " if r["true_label"] == r["pred_label"] else "WRONG"
        probs_str = "  ".join(
            f"{class_names[j]}:{r['probs'][j]*100:.1f}%"
            for j in range(len(class_names))
        )
        print(f"  [{i+1:>2}] {status}  True:{true_name:<12} "
              f"Pred:{pred_name:<12} ({r['confidence']:.1f}%)  |  {probs_str}")

    # Plot
    print(f"\n  Generating 4x4 visualisation grid ...")
    plot_sanity_grid(results, class_names)

    print("\n" + "=" * 60)
    print("  PHASE 2B COMPLETE")
    print(f"  Output : {SAVE_PATH}")
    print("  If soft-label bars show non-trivial probability spread")
    print("  across classes, the teacher is producing good dark knowledge")
    print("  for Phase 3 distillation.")
    print("=" * 60)


if __name__ == "__main__":
    main()