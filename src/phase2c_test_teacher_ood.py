"""
=============================================================================
  3D PRINTING DEFECT DETECTION — OOD TEST
  test_internet_image.py
=============================================================================
  Downloads an image from a URL, preprocesses it identically to the val
  set, and runs it through the trained MobileNetV4-Medium teacher.
  Prints Top-1 prediction + full softmax probability breakdown.
=============================================================================
"""

import sys
import os
from io import BytesIO

import requests
from PIL import Image

import torch
import torch.nn as nn
import timm
from torchvision.transforms import v2

# ── Phase 1 imports ──────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase1_data_pipeline import IMAGENET_MEAN, IMAGENET_STD

# =============================================================================
#  CONFIGURATION  — swap IMAGE_URL for any public image link
# =============================================================================

IMAGE_URL       = "https://cdn.mos.cms.futurecdn.net/TAKexdhQDGcVVmaVJMxHxh-650-80.jpg.webp"  # ← change this
TEACHER_WEIGHTS = "outputs/models/teacher_best.pth"
NUM_CLASSES     = 3
IMAGE_SIZE      = 224
CLASS_NAMES     = ["spaghetti", "stringing", "zits"]   # must match training order
MODEL_NAME      = "mobilenetv4_conv_medium"

# Request headers — some hosts block requests without a user-agent
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# =============================================================================
#  1.  IMAGE FETCHING
# =============================================================================

def fetch_image(url: str) -> Image.Image:
    """
    Download image from URL and return a PIL Image in RGB mode.
    Raises with a clear message on network or decode errors.
    """
    print(f"  Fetching image from:\n  {url}\n")
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()         # raises on 4xx / 5xx
    except requests.exceptions.ConnectionError:
        print("  ERROR: Could not connect. Check the URL or your internet.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"  ERROR: HTTP {e.response.status_code} — server rejected request.")
        print("  Try a direct image link (ending in .jpg / .png).")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("  ERROR: Request timed out after 15 seconds.")
        sys.exit(1)

    try:
        img = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"  ERROR: Could not decode image — {e}")
        print("  Make sure the URL points directly to an image file.")
        sys.exit(1)

    print(f"  Downloaded OK  |  Original size : {img.width} x {img.height} px")
    return img


# =============================================================================
#  2.  PREPROCESSING  (identical to val/test — no augmentations)
# =============================================================================

def build_eval_transform() -> v2.Compose:
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


def preprocess(img: Image.Image, transform: v2.Compose) -> torch.Tensor:
    """Returns a (1, 3, 224, 224) float32 tensor ready for the model."""
    tensor = transform(img)          # (3, 224, 224)
    tensor = tensor.unsqueeze(0)     # (1, 3, 224, 224)
    return tensor


# =============================================================================
#  3.  MODEL LOADING
# =============================================================================

def load_teacher(weights_path: str, device: torch.device) -> nn.Module:
    if not os.path.isfile(weights_path):
        print(f"  ERROR: Weights file not found — '{weights_path}'")
        print("  Run phase2_train_teacher.py first.")
        sys.exit(1)

    model = timm.create_model(
        MODEL_NAME,
        pretrained  = False,
        num_classes = NUM_CLASSES,
    )
    state_dict = torch.load(
        weights_path,
        map_location = device,
        weights_only = True,    # safe load — avoids arbitrary code execution
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"  Weights loaded  : {weights_path}")
    print(f"  Device          : {device}")
    return model


# =============================================================================
#  4 + 5.  INFERENCE + OUTPUT
# =============================================================================

@torch.no_grad()
def run_inference(
    model:   nn.Module,
    tensor:  torch.Tensor,
    device:  torch.device,
):
    tensor = tensor.to(device)
    logits = model(tensor)                              # (1, 3)
    probs  = torch.softmax(logits, dim=1).squeeze(0)   # (3,)

    top1_idx   = probs.argmax().item()
    top1_class = CLASS_NAMES[top1_idx]
    top1_conf  = probs[top1_idx].item() * 100

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  INFERENCE RESULT")
    print("=" * 52)

    # Confidence bar (40 chars wide)
    bar_len = int(top1_conf / 100 * 40)
    bar     = "█" * bar_len + "░" * (40 - bar_len)

    print(f"\n  Top-1 Prediction : {top1_class.upper()}")
    print(f"  Confidence       : {top1_conf:.2f}%")
    print(f"  [{bar}]")

    print("\n  Full probability breakdown:")
    print("  " + "-" * 38)
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs.tolist())):
        marker   = " <-- predicted" if i == top1_idx else ""
        mini_bar = "█" * int(prob * 30)
        print(f"  {name:<12} {prob*100:>6.2f}%  {mini_bar}{marker}")

    print("\n" + "=" * 52)

    # Interpretation hint
    if top1_conf >= 90:
        print(f"  High confidence — teacher is certain this is {top1_class}.")
    elif top1_conf >= 60:
        print(f"  Moderate confidence — some ambiguity between classes.")
        runner_up_idx  = probs.topk(2).indices[1].item()
        runner_up_name = CLASS_NAMES[runner_up_idx]
        runner_up_conf = probs[runner_up_idx].item() * 100
        print(f"  Runner-up: {runner_up_name} ({runner_up_conf:.2f}%)")
    else:
        print("  Low confidence — image may be OOD or a borderline case.")
        print("  This is expected for images very different from the training set.")

    print("=" * 52)

    return top1_class, top1_conf, probs


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 52)
    print("  OOD INTERNET IMAGE TEST")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Classes: {CLASS_NAMES}")
    print("=" * 52 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Fetch
    img = fetch_image(IMAGE_URL)

    # 2. Preprocess
    transform = build_eval_transform()
    tensor    = preprocess(img, transform)
    print(f"  Tensor shape    : {tuple(tensor.shape)}")
    print(f"  Tensor dtype    : {tensor.dtype}")

    # 3. Load model
    model = load_teacher(TEACHER_WEIGHTS, device)

    # 4 + 5. Infer + print results
    run_inference(model, tensor, device)


if __name__ == "__main__":
    main()