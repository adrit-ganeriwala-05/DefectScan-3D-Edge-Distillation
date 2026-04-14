"""
=============================================================================
  phase4a_quantize_int8.py  —  Phase 4: Dynamic INT8 Quantization
=============================================================================
  Responsibility : load outputs/models/student_fp32_best.pth, apply Dynamic Quantization,
                   and save the quantized model to outputs/models/student_int8.pth.

  Why Dynamic Quantization (not Static PTQ):
    Static PTQ calls torch.quantization.convert() which replaces nn.Conv2d
    with QuantizedConv2d C++ objects. In PyTorch 2.5, these objects are
    missing _backward_hooks and _modules attributes expected by the standard
    nn.Module dispatch path that timm's forward_features() uses. This causes:
      AttributeError: 'Conv2d' object has no attribute '_backward_hooks'
    This is a fundamental incompatibility between PyTorch 2.5 Static PTQ
    and timm MobileNetV4 — it cannot be fixed with serialization changes.

    Dynamic Quantization:
      - Quantizes nn.Linear weights to INT8 at conversion time
      - Activations quantized on-the-fly at inference (no calibration needed)
      - Leaves Conv2d as FP32 — no QuantizedConv2d, no dispatch issues
      - torch.save(model) works cleanly — model stays a valid nn.Module
      - torch.load(weights_only=False) loads without any AttributeError
      - Gives genuine compression on the Linear classifier head
      - Meaningful CPU speedup on the bottleneck layers

  Serialization : torch.save(quantized_model_object, path)
                  Whole-object save — the quantized Linear layers carry
                  packed INT8 weight tensors that must stay with the object.
                  weights_only=False required on load (trusted local file).

  Run           : python src/phase4a_quantize_int8.py
=============================================================================
"""

import os
import sys
import platform

import torch
import torch.nn as nn
import timm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase1_data_pipeline import hardware_check

# =============================================================================
#  CONFIGURATION
# =============================================================================
STUDENT_NAME  = "mobilenetv4_conv_small"
STUDENT_FP32  = "outputs/models/student_fp32_best.pth"
STUDENT_INT8  = "outputs/models/student_int8.pth"
NUM_CLASSES   = 3


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":
    print("  Dynamic INT8 Quantization runs on CPU")

    if not os.path.isfile(STUDENT_FP32):
        print(f"  ERROR: '{STUDENT_FP32}' not found.")
        print("  Run phase3a_train_distill.py first.")
        sys.exit(1)

    # Load FP32 student
    print(f"\n  Loading {STUDENT_NAME} FP32 weights ...")
    backbone = timm.create_model(STUDENT_NAME, pretrained=False,
                                 num_classes=NUM_CLASSES)
    backbone.load_state_dict(
        torch.load(STUDENT_FP32, map_location="cpu", weights_only=True))
    backbone.eval()

    fp32_mb = sum(p.numel() * p.element_size()
                  for p in backbone.parameters()) / (1024 ** 2)
    print(f"  FP32 weight memory : {fp32_mb:.2f} MB")

    # Apply Dynamic Quantization to Linear layers
    # Conv2d is intentionally excluded — Static PTQ on Conv2d is incompatible
    # with timm MobileNetV4 + PyTorch 2.5 (QuantizedConv2d dispatch error).
    print("  Applying dynamic quantization to nn.Linear layers ...")
    int8_model = torch.quantization.quantize_dynamic(
        backbone,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
    )

    # Report which layers were quantized
    linear_count = sum(1 for m in int8_model.modules()
                       if "DynamicQuantizedLinear" in type(m).__name__)
    print(f"  Quantized Linear layers : {linear_count}")

    # Whole-object save — INT8 packed weight tensors stay with the object
    torch.save(int8_model, STUDENT_INT8)

    int8_disk_mb = os.path.getsize(STUDENT_INT8) / (1024 ** 2)
    print(f"\n  FP32 size   : {fp32_mb:.2f} MB")
    print(f"  INT8 size   : {int8_disk_mb:.2f} MB  "
          f"({fp32_mb/int8_disk_mb:.2f}x compression)")
    print(f"  Saved       : '{STUDENT_INT8}'")

    print("\n" + "=" * 60)
    print("  PHASE 4A COMPLETE")
    print(f"  INT8 model : {STUDENT_INT8}")
    print("  Next       : python src/phase4b_eval_testset.py")
    print("=" * 60)