"""
=============================================================================
  phase4b_eval_testset.py  —  Phase 4: INT8 Student Test Set Evaluation
                               + Quantization Drop Report
                               + Edge Simulation Benchmark
=============================================================================
  Responsibility:
    1. Load outputs/models/student_int8.pth via whole-object deserialization.
    2. Evaluate on the held-out test set (accuracy, Macro F1, report,
       confusion matrix PNG).
    3. Load phase3b_results.json and compute Quantization Drop.
    4. Report memory footprint (Teacher FP32 / Student FP32 / Student INT8).
    5. Run edge simulation benchmark in a subprocess-isolated worker:
         - CUDA_VISIBLE_DEVICES="-1" injected via env= (hard isolation)
         - 4 intra-op / 1 inter-op CPU threads
         - Teacher FP32 with channels_last (fairness)
         - Student INT8 in default contiguous format
         - Reports: avg/P50/P95 latency, std dev, FPS, throughput

  INT8 loading rules:
    - torch.load(path, map_location="cpu", weights_only=False)
    - Never use timm.create_model for the INT8 model.
    - No .eval() — DynamicQuantizedLinear model is already in inference mode.
    - FutureWarning suppressed — trusted phase4a artifact.

  Run: python src/phase4b_eval_testset.py
=============================================================================
"""

import os
import sys
import json
import time
import platform
import subprocess
import tempfile
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
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
TEACHER_NAME     = "mobilenetv4_conv_medium"
STUDENT_NAME     = "mobilenetv4_conv_small"
TEACHER_WEIGHTS  = "outputs/models/teacher_best.pth"
STUDENT_FP32     = "outputs/models/student_fp32_best.pth"
STUDENT_INT8     = "outputs/models/student_int8.pth"
PHASE3B_RESULTS  = "phase3b_results.json"
NUM_CLASSES      = 3
BATCH_SIZE       = 64
CONFUSION_PATH   = "phase4b_student_int8_confusion_matrix.png"
BENCHMARK_PATH   = "phase4b_benchmark.png"
RESULTS_JSON     = "phase4b_results.json"

EDGE_CPU_THREADS = 4
EDGE_INTEROP     = 1
BENCHMARK_N      = 200
WORKER_FLAG      = "--phase4b-benchmark-worker"


# =============================================================================
#  HELPERS
# =============================================================================

def detect_quant_backend() -> str:
    machine = platform.machine().lower()
    backend = "qnnpack" if machine in ("arm", "aarch64", "arm64") else "fbgemm"
    print(f"  Quantization backend : {backend}  (arch: {machine})")
    return backend


def eval_transform() -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE),
                  interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_int8_model() -> nn.Module:
    """
    Whole-object load for the Dynamic Quantization model produced by phase4a.

    weights_only=False is required — the model object contains
    DynamicQuantizedLinear layers with packed INT8 weight tensors that
    are not plain state_dict entries. FutureWarning is suppressed because
    this is a trusted artifact from our own pipeline.

    No .eval() needed — dynamically quantized models are already in
    inference mode after quantize_dynamic().
    """
    if not os.path.isfile(STUDENT_INT8):
        print(f"  ERROR: '{STUDENT_INT8}' not found.")
        print("  Run phase4a_quantize_int8.py first.")
        sys.exit(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        # weights_only=False required: trusted phase4a artifact (whole object)
        model = torch.load(STUDENT_INT8, map_location="cpu",
                           weights_only=False)
    return model


def save_confusion_matrix(labels: np.ndarray, preds: np.ndarray,
                           class_names: list, acc: float) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#0f0f23")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_title(f"Student INT8 (Dynamic) — Test Set\nAcc: {acc:.2f}%",
                 color="white", fontsize=11, pad=12)
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("True",      color="white")
    ax.tick_params(colors="white")
    plt.setp(ax.get_xticklabels(), color="white", rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), color="white", rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, bbox_inches="tight",
                dpi=120, facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Confusion matrix -> '{CONFUSION_PATH}'")


def print_quant_drop(int8_acc: float, int8_f1: float) -> None:
    if not os.path.isfile(PHASE3B_RESULTS):
        print(f"\n  [SKIP] '{PHASE3B_RESULTS}' not found — "
              "run phase3b_eval_testset.py first.")
        return
    with open(PHASE3B_RESULTS) as f:
        r = json.load(f)
    fp32_acc = r.get("accuracy")
    fp32_f1  = r.get("macro_f1")
    if fp32_acc is None:
        print("\n  [SKIP] phase3b_results.json missing 'accuracy' key.")
        return

    drop_acc = fp32_acc - int8_acc
    print("\n" + "=" * 60)
    print("  QUANTIZATION DROP REPORT  (FP32 → Dynamic INT8)")
    print("=" * 60)
    print(f"  FP32 Accuracy   : {fp32_acc:.2f}%")
    print(f"  INT8 Accuracy   : {int8_acc:.2f}%")
    print(f"  Drop (Accuracy) : {drop_acc:.2f} pp")
    if fp32_f1 is not None:
        print(f"  FP32 Macro F1   : {fp32_f1:.4f}")
        print(f"  INT8 Macro F1   : {int8_f1:.4f}")
        print(f"  Drop (F1)       : {fp32_f1 - int8_f1:.4f}")
    if   drop_acc <= 1.0: v = "[EXCELLENT]   <=1 pp — production-ready"
    elif drop_acc <= 3.0: v = "[ACCEPTABLE]  <=3 pp — usable for edge"
    elif drop_acc <= 5.0: v = "[MARGINAL]    <=5 pp — acceptable for most cases"
    else:                 v = "[NOTABLE]     >5 pp — expected for Linear-only quant"
    print(f"\n  {v}")
    print("  Note: Dynamic quant quantizes Linear layers only (Conv2d=FP32)")
    print("=" * 60)


def print_memory_footprint(int8_model: nn.Module) -> None:
    print("\n" + "=" * 60)
    print("  MEMORY FOOTPRINT")
    print("=" * 60)
    rows = []

    def _mb(m):
        return sum(p.numel() * p.element_size()
                   for p in m.parameters()) / (1024 ** 2)

    if os.path.isfile(TEACHER_WEIGHTS):
        t = timm.create_model(TEACHER_NAME, pretrained=False,
                               num_classes=NUM_CLASSES)
        t.load_state_dict(torch.load(TEACHER_WEIGHTS, map_location="cpu",
                                     weights_only=True))
        rows.append(("Teacher  FP32",
                     os.path.getsize(TEACHER_WEIGHTS) / (1024**2), _mb(t)))
        del t

    if os.path.isfile(STUDENT_FP32):
        s = timm.create_model(STUDENT_NAME, pretrained=False,
                               num_classes=NUM_CLASSES)
        s.load_state_dict(torch.load(STUDENT_FP32, map_location="cpu",
                                     weights_only=True))
        rows.append(("Student  FP32",
                     os.path.getsize(STUDENT_FP32) / (1024**2), _mb(s)))
        del s

    rows.append(("Student  INT8 (dyn)",
                 os.path.getsize(STUDENT_INT8) / (1024**2),
                 _mb(int8_model)))

    print(f"\n  {'Model':<22} {'Disk (MB)':>10}  {'RAM Weights':>12}")
    print("  " + "-" * 48)
    for name, disk, ram in rows:
        print(f"  {name:<22} {disk:>10.2f}  {ram:>10.2f} MB")

    if len(rows) == 3:
        saving = rows[0][2] - rows[2][2]
        print(f"\n  RAM saving (Teacher→INT8) : "
              f"{saving:.2f} MB  ({saving/rows[0][2]*100:.1f}%)")
    print("=" * 60)


# =============================================================================
#  BENCHMARK WORKER
#  CUDA isolation: CUDA_VISIBLE_DEVICES="-1" set via env= in subprocess.run(),
#  NOT inside this function — present before Python interpreter starts.
# =============================================================================

def _benchmark_worker() -> None:
    torch.set_num_threads(EDGE_CPU_THREADS)
    torch.set_num_interop_threads(EDGE_INTEROP)

    cuda_visible   = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")
    cuda_available = torch.cuda.is_available()

    print(f"\n  [Worker] PID                  : {os.getpid()}")
    print(f"  [Worker] CUDA_VISIBLE_DEVICES : {cuda_visible}")
    print(f"  [Worker] torch.cuda.available : {cuda_available}")
    print(f"  [Worker] intra-op threads     : {torch.get_num_threads()}")
    print(f"  [Worker] inter-op threads     : {torch.get_num_interop_threads()}")
    if not cuda_available and cuda_visible == "-1":
        print("  [Worker] Isolation            : VERIFIED ✓")
    else:
        print("  [Worker] Isolation            : FAILED ✗")

    teacher_path  = sys.argv[2]
    int8_path     = sys.argv[3]
    result_path   = sys.argv[4]
    dataset_root  = sys.argv[5]

    print(f"\n  [Worker] Loading {BENCHMARK_N} test images ...")
    tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE),
                  interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_ds = tvdatasets.ImageFolder(
        root=os.path.join(dataset_root, "test"), transform=tf)
    total = len(test_ds)
    step  = max(1, total // BENCHMARK_N)
    imgs  = [test_ds[min(i * step, total - 1)][0].unsqueeze(0)
             for i in range(BENCHMARK_N)]
    print(f"  [Worker] {len(imgs)} images ready (from {total})")

    def _bench(model, label):
        model.eval()
        lats = []
        with torch.no_grad():
            for img in imgs[:10]:
                _ = model(img)
            for img in imgs:
                t0 = time.perf_counter()
                _  = model(img)
                lats.append((time.perf_counter() - t0) * 1000)
        avg = float(np.mean(lats))
        p50 = float(np.percentile(lats, 50))
        p95 = float(np.percentile(lats, 95))
        std = float(np.std(lats))
        fps = 1000.0 / avg
        print(f"\n  [{label}] avg={avg:.2f}ms p50={p50:.2f}ms "
              f"p95={p95:.2f}ms std={std:.2f}ms fps={fps:.1f}")
        return {"label": label, "avg": avg, "p50": p50, "p95": p95,
                "std": std, "fps": fps, "throughput": fps, "latencies": lats}

    # Teacher FP32 — channels_last for fairness
    print("\n  [Worker] Teacher FP32 (channels_last) ...")
    t_bb = timm.create_model(TEACHER_NAME, pretrained=False,
                              num_classes=NUM_CLASSES)
    t_bb.load_state_dict(
        torch.load(teacher_path, map_location="cpu", weights_only=True))
    t_bb = t_bb.to(memory_format=torch.channels_last)
    r_teacher = _bench(t_bb, "Teacher FP32 (channels_last)")

    # Student INT8 — whole-object load, contiguous format
    print("\n  [Worker] Student INT8 (dynamic quant, whole-object) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        int8_model = torch.load(int8_path, map_location="cpu",
                                weights_only=False)
    r_student = _bench(int8_model, "Student INT8 (dynamic)")

    with open(result_path, "w") as f:
        json.dump({"teacher": r_teacher, "student": r_student}, f)
    print(f"\n  [Worker] Done -> {result_path}")


# =============================================================================
#  BENCHMARK LAUNCHER
# =============================================================================

def run_benchmark() -> tuple[dict, dict]:
    print("\n" + "=" * 65)
    print("  EDGE SIMULATION BENCHMARK  (subprocess-isolated)")
    print(f"  {EDGE_CPU_THREADS} intra-op | {EDGE_INTEROP} inter-op | "
          f"CUDA disabled | N={BENCHMARK_N}")
    print("=" * 65)

    for path in [TEACHER_WEIGHTS, STUDENT_INT8]:
        if not os.path.isfile(path):
            print(f"  ERROR: '{path}' not found.")
            return {}, {}

    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False) as tmp:
        result_path = tmp.name

    cmd = [
        sys.executable, os.path.abspath(__file__),
        WORKER_FLAG,
        os.path.abspath(TEACHER_WEIGHTS),
        os.path.abspath(STUDENT_INT8),
        result_path,
        os.path.abspath(DATASET_ROOT),
    ]
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = "-1"

    print("\n  Spawning worker ...")
    proc = subprocess.run(cmd, check=False, env=child_env)

    if proc.returncode != 0:
        print(f"\n  ERROR: worker exited {proc.returncode}")
        return {}, {}

    with open(result_path) as f:
        results = json.load(f)
    os.unlink(result_path)

    r_t = results["teacher"]
    r_s = results["student"]
    speedup = r_t["avg"] / r_s["avg"]

    print("\n" + "=" * 65)
    print("  BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"  {'Metric':<28} {'Teacher FP32':>14}  {'Student INT8':>14}")
    print("  " + "-" * 58)
    for key, lbl in [("avg", "Avg Latency (ms)"), ("p50", "P50 (ms)"),
                     ("p95", "P95 (ms)"), ("std", "Std Dev / Jitter (ms)"),
                     ("fps", "FPS"), ("throughput", "Throughput (img/s)")]:
        print(f"  {lbl:<28} {r_t[key]:>14.2f}  {r_s[key]:>14.2f}")
    print("  " + "-" * 58)
    print(f"  Speedup : {speedup:.2f}x  |  "
          f"FPS gain : {r_s['fps']/r_t['fps']:.2f}x")
    stability = "Student" if r_s["std"] < r_t["std"] else "Teacher"
    diff = abs(r_t["std"] - r_s["std"])
    print(f"  Jitter  : {stability} is {diff:.2f} ms more stable")
    print("=" * 65)

    _plot_benchmark(r_t, r_s, speedup)
    return r_t, r_s


def _plot_benchmark(r_t: dict, r_s: dict, speedup: float) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), facecolor="#1a1a2e")
    fig.suptitle(
        f"Phase 4B — Edge Benchmark  "
        f"({EDGE_CPU_THREADS}-thread CPU · CUDA disabled · "
        f"N={BENCHMARK_N} · Dynamic INT8)",
        color="white", fontsize=9)

    def _style(ax):
        ax.set_facecolor("#0f0f23")
        ax.tick_params(colors="white")
        ax.grid(alpha=0.2, axis="y")
        for s in ax.spines.values():
            s.set_edgecolor("#444")

    ax = axes[0]
    ax.set_facecolor("#0f0f23")
    ax.hist(r_t["latencies"], bins=25, color="#ef5350", alpha=0.7,
            label=f"Teacher\nμ={r_t['avg']:.1f} σ={r_t['std']:.1f}ms")
    ax.hist(r_s["latencies"], bins=25, color="#4fc3f7", alpha=0.7,
            label=f"INT8\nμ={r_s['avg']:.1f} σ={r_s['std']:.1f}ms")
    ax.set_title("Latency Distribution", color="white")
    ax.set_xlabel("ms", color="white")
    ax.set_ylabel("Count", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7)
    ax.grid(alpha=0.2)
    for s in ax.spines.values(): s.set_edgecolor("#444")

    cfgs = [
        ([r_t["avg"],        r_s["avg"]],       "Avg Latency",  "ms",    "{:.1f}", False),
        ([r_t["fps"],        r_s["fps"]],        f"FPS ({speedup:.2f}x)", "FPS", "{:.1f}", False),
        ([r_t["throughput"], r_s["throughput"]], "Throughput",   "img/s", "{:.1f}", False),
        ([r_t["std"],        r_s["std"]],        "Jitter (Std)", "ms",   "{:.2f}", True),
    ]
    for ax, (vals, title, ylabel, fmt, inv) in zip(axes[1:], cfgs):
        cols = ["#ef5350", "#69f0ae"] if inv else ["#ef5350", "#4fc3f7"]
        bars = ax.bar(["Teacher\nFP32", "Student\nINT8"],
                      vals, color=cols, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.015,
                    fmt.format(v), ha="center", color="white",
                    fontsize=9, fontweight="bold")
        ax.set_title(title, color="white", fontsize=9)
        ax.set_ylabel(ylabel, color="white")
        _style(ax)

    plt.tight_layout()
    plt.savefig(BENCHMARK_PATH, bbox_inches="tight",
                dpi=120, facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Benchmark plot -> '{BENCHMARK_PATH}'")


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == WORKER_FLAG:
        _benchmark_worker()
        sys.exit(0)

    hardware_check()
    detect_quant_backend()

    test_ds     = tvdatasets.ImageFolder(
        root=os.path.join(DATASET_ROOT, "test"),
        transform=eval_transform())
    class_names = test_ds.classes
    print(f"  Classes     : {class_names}")
    print(f"  Test images : {len(test_ds)}")

    # Load INT8 — whole-object only, never timm.create_model
    print(f"\n  Loading Student INT8 ...")
    int8_model = load_int8_model()

    # Inference on test set — CPU only, no channels_last
    loader = DataLoader(
        tvdatasets.ImageFolder(root=os.path.join(DATASET_ROOT, "test"),
                               transform=eval_transform()),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = int8_model(images).argmax(dim=1).numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc        = float((all_preds == all_labels).mean() * 100)
    macro_f1   = float(f1_score(all_labels, all_preds,
                                average="macro", zero_division=0))

    print("\n" + "=" * 60)
    print("  STUDENT INT8 (DYNAMIC) TEST SET RESULTS")
    print("=" * 60)
    print(f"  Accuracy : {acc:.2f}%")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print("\n  Classification Report:")
    print("  " + "-" * 52)
    for line in classification_report(
        all_labels, all_preds, target_names=class_names, digits=4,
    ).splitlines():
        print(f"  {line}")

    save_confusion_matrix(all_labels, all_preds, class_names, acc)
    print_quant_drop(acc, macro_f1)
    print_memory_footprint(int8_model)
    run_benchmark()

    with open(RESULTS_JSON, "w") as f:
        json.dump({"model": STUDENT_NAME + "_int8_dynamic",
                   "accuracy": acc, "macro_f1": macro_f1}, f, indent=2)
    print(f"\n  Results JSON -> '{RESULTS_JSON}'")

    print("\n" + "=" * 60)
    print("  PHASE 4B COMPLETE")
    print(f"  Accuracy  : {acc:.2f}%  |  Macro F1: {macro_f1:.4f}")
    print(f"  Confusion : {CONFUSION_PATH}")
    print(f"  Benchmark : {BENCHMARK_PATH}")
    print("=" * 60)