"""
=============================================================================
  phase4c_eval_ood.py  —  Phase 4: INT8 Student OOD Single-Image Inference
=============================================================================
    Responsibility : run a single "wild" image through outputs/models/student_int8.pth and
                   display top-3 class probabilities.

  Loading rule   : whole-object deserialization only.
                   torch.load(path, map_location="cpu", weights_only=False)
                   Never use timm.create_model for the INT8 model.

  Usage  : python src/phase4c_eval_ood.py <image_path_or_url>
  Example: python src/phase4c_eval_ood.py https://example.com/print.jpg
           python src/phase4c_eval_ood.py /path/to/image.jpg
=============================================================================
"""

import os
import sys
import io
import platform
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase1_data_pipeline import (
    hardware_check, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE,
)

# =============================================================================
#  CONFIGURATION
# =============================================================================
STUDENT_INT8    = "outputs/models/student_int8.pth"
CLASS_NAMES     = ["spaghetti", "stringing", "zits"]
OOD_OUTPUT_PATH = "phase4c_ood_result.png"


# =============================================================================
#  HELPERS
# =============================================================================

def detect_quant_backend() -> str:
    machine = platform.machine().lower()
    backend = "qnnpack" if machine in ("arm", "aarch64", "arm64") else "fbgemm"
    print(f"  Quantization backend : {backend}  (arch: {machine})")
    return backend


def load_int8_model() -> torch.nn.Module:
    """
    Whole-object deserialization — the only correct load path for a
    torch.quantization.convert() model. weights_only=False is required
    because QuantizedConv2d objects are not plain tensors. FutureWarning
    suppressed — trusted phase4a artifact.
    """
    if not os.path.isfile(STUDENT_INT8):
        print(f"  ERROR: '{STUDENT_INT8}' not found.")
        print("  Run phase4a_quantize_int8.py first.")
        sys.exit(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        model = torch.load(STUDENT_INT8, map_location="cpu",
                           weights_only=False)
    # Do NOT call .eval() — raises AttributeError on QuantizedConv2d
    return model


def load_image(source: str) -> Image.Image:
    if source.startswith("http://") or source.startswith("https://"):
        import urllib.request
        req = urllib.request.Request(source, headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    if not os.path.isfile(source):
        print(f"  ERROR: File not found: '{source}'")
        sys.exit(1)
    return Image.open(source).convert("RGB")


def build_transform() -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE),
                  interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def save_result_chart(probs: np.ndarray, pred_class: str,
                      pred_conf: float) -> None:
    colours = ["#69f0ae" if i == int(np.argmax(probs)) else "#4fc3f7"
               for i in range(len(CLASS_NAMES))]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#0f0f23")
    bars = ax.barh(CLASS_NAMES, probs * 100, color=colours)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", color="white", fontsize=10)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", color="white")
    ax.set_title(
        f"Student INT8 OOD Inference\n"
        f"Prediction: {pred_class}  ({pred_conf*100:.1f}%)",
        color="white", fontsize=11)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_edgecolor("#444")
    ax.grid(alpha=0.2, axis="x")
    plt.tight_layout()
    plt.savefig(OOD_OUTPUT_PATH, bbox_inches="tight",
                dpi=120, facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Result chart -> '{OOD_OUTPUT_PATH}'")


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/phase4c_eval_ood.py <image_path_or_url>")
        sys.exit(1)

    source = sys.argv[1]
    hardware_check()
    quant_backend = detect_quant_backend()

    print(f"\n  Loading Student INT8 ...")
    torch.backends.quantized.engine = quant_backend
    model = load_int8_model()

    print(f"  Loading image: {source}")
    image  = load_image(source)
    tensor = build_transform()(image).unsqueeze(0)
    # No channels_last — quantized ops do not support NHWC

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    pred_conf  = float(probs[pred_idx])

    print("\n" + "=" * 55)
    print("  STUDENT INT8 OOD RESULT")
    print("=" * 55)
    for i, (cls, p) in enumerate(zip(CLASS_NAMES, probs)):
        marker = "  <-- PREDICTION" if i == pred_idx else ""
        print(f"  {cls:<20} {p*100:>6.2f}%{marker}")
    print("=" * 55)
    print(f"  Prediction : {pred_class}  ({pred_conf*100:.1f}% confidence)")

    save_result_chart(probs, pred_class, pred_conf)
    print("\n  PHASE 4C COMPLETE")