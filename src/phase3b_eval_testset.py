"""
=============================================================================
  phase3b_eval_testset.py  —  Phase 3: FP32 Student Test Set Evaluation
=============================================================================
    Responsibility : evaluate outputs/models/student_fp32_best.pth on the held-out test set.
  Outputs        : accuracy, macro F1, classification report,
                   confusion matrix PNG, phase3b_results.json.
  Run            : python src/phase3b_eval_testset.py
=============================================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
import timm
from torchvision import datasets as tvdatasets
from torchvision.transforms import v2
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase1_data_pipeline import (
    hardware_check, DATASET_ROOT, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE,
)

# =============================================================================
#  CONFIGURATION
# =============================================================================
STUDENT_NAME    = "mobilenetv4_conv_small"
STUDENT_FP32    = "outputs/models/student_fp32_best.pth"
NUM_CLASSES     = 3
BATCH_SIZE      = 64
NUM_WORKERS     = 4
CONFUSION_PATH  = "phase3b_student_fp32_confusion_matrix.png"
RESULTS_JSON    = "phase3b_results.json"   # read by phase4b for quant-drop report


# =============================================================================
#  HELPERS
# =============================================================================

def eval_transform() -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE),
                  interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def save_confusion_matrix(labels: np.ndarray, preds: np.ndarray,
                           class_names: list, acc: float,
                           title: str, path: str) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#0f0f23")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_title(f"{title}\nAcc: {acc:.2f}%", color="white", fontsize=11, pad=12)
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("True", color="white")
    ax.tick_params(colors="white")
    plt.setp(ax.get_xticklabels(), color="white", rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), color="white", rotation=0)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=120, facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Confusion matrix -> '{path}'")


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":
    device = hardware_check()

    if not os.path.isfile(STUDENT_FP32):
        print(f"  ERROR: '{STUDENT_FP32}' not found.")
        print("  Run phase3a_train_distill.py first.")
        sys.exit(1)

    print(f"\n  Loading {STUDENT_NAME} ...")
    model = timm.create_model(STUDENT_NAME, pretrained=False,
                              num_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load(STUDENT_FP32, map_location=device, weights_only=True))
    model = model.to(memory_format=torch.channels_last).to(device)
    model.eval()

    test_ds     = tvdatasets.ImageFolder(
        root=os.path.join(DATASET_ROOT, "test"), transform=eval_transform())
    class_names = test_ds.classes
    print(f"  Classes     : {class_names}")
    print(f"  Test images : {len(test_ds)}")

    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True,
                               memory_format=torch.channels_last)
            preds  = model(images).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc        = float((all_preds == all_labels).mean() * 100)
    macro_f1   = float(f1_score(all_labels, all_preds,
                                average="macro", zero_division=0))

    print("\n" + "=" * 60)
    print("  STUDENT FP32 TEST SET RESULTS")
    print("=" * 60)
    print(f"  Accuracy : {acc:.2f}%")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print("\n  Classification Report:")
    print("  " + "-" * 52)
    for line in classification_report(
        all_labels, all_preds, target_names=class_names, digits=4,
    ).splitlines():
        print(f"  {line}")

    save_confusion_matrix(all_labels, all_preds, class_names, acc,
                          "Student FP32 — Test Set", CONFUSION_PATH)

    results = {"model": STUDENT_NAME, "accuracy": acc, "macro_f1": macro_f1}
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON -> '{RESULTS_JSON}'")

    print("\n" + "=" * 60)
    print("  PHASE 3B COMPLETE")
    print(f"  Accuracy : {acc:.2f}%  |  Macro F1: {macro_f1:.4f}")
    print(f"  Run phase4a_quantize_int8.py to produce outputs/models/student_int8.pth")
    print("=" * 60)