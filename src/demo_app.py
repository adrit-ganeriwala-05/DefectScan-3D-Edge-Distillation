"""
=============================================================================
  demo_app.py  —  DefectScan AI  |  3D Print Defect Detector
=============================================================================
  Final fixed version
  - Upload text overlap fixed for Streamlit 1.56.0
  - Defect Reference section cleaned up
  - About This Model section rebuilt so it renders correctly
=============================================================================
"""

import os
import io
import tempfile
from PIL import Image

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.transforms import v2

import streamlit as st


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="DefectScan AI",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# CONSTANTS
# =============================================================================
STUDENT_NAME = "mobilenetv4_conv_small"
WEIGHTS_PATH = "outputs/models/student_fp32_best.pth"
NUM_CLASSES = 3
CLASS_NAMES = ["spaghetti", "stringing", "zits"]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_META = {
    "spaghetti": {
        "icon": "⚠️",
        "hex": "#e53935",
        "severity": "CRITICAL",
        "action": "Stop print immediately",
        "description": (
            "Filament has collapsed and is producing tangled strands. "
            "The print has failed and must be restarted."
        ),
    },
    "stringing": {
        "icon": "🔶",
        "hex": "#ff7043",
        "severity": "MODERATE",
        "action": "Adjust retraction settings",
        "description": (
            "Fine threads left between features during travel moves. "
            "Reduce travel speed or increase retraction distance."
        ),
    },
    "zits": {
        "icon": "🔴",
        "hex": "#e57373",
        "severity": "MINOR",
        "action": "Review extrusion multiplier",
        "description": (
            "Surface blobs from excess extrusion. "
            "Slightly reduce flow rate or check for partial clogs."
        ),
    },
}


# =============================================================================
# THEME
# =============================================================================
def T(key: str) -> str:
    dark = {
        "bg": "#0a0a0a",
        "bg2": "#111111",
        "bg3": "#1a1a1a",
        "border": "#242424",
        "text1": "#f0f0f0",
        "text2": "#a0a0a0",
        "text3": "#505050",
        "accent": "#e53935",
        "bar_bg": "#1e1e1e",
        "card": "#111111",
        "success_bg": "#0d2b0d",
        "success_text": "#81c784",
        "warn_bg": "#2b1a00",
        "warn_text": "#ffcc80",
        "info_bg": "#0d1f2b",
        "info_text": "#81d4fa",
    }
    return dark[key]


# =============================================================================
# CSS
# =============================================================================
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Rajdhani:wght@600;700&display=swap');

        html, body, .stApp {{
            background-color: {T('bg')} !important;
            font-family: 'Inter', sans-serif !important;
            color: {T('text1')} !important;
        }}

        .main .block-container {{
            max-width: 1100px !important;
            padding: 0 2rem 4rem 2rem !important;
        }}

        p, li, span, label {{
            color: {T('text2')} !important;
            font-family: 'Inter', sans-serif !important;
        }}

        h1, h2, h3, h4, h5 {{
            color: {T('text1')} !important;
            font-family: 'Rajdhani', sans-serif !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            background-color: {T('bg3')} !important;
            border-radius: 8px !important;
            padding: 4px !important;
            gap: 2px !important;
            border: none !important;
        }}

        .stTabs [data-baseweb="tab"] {{
            color: {T('text2')} !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            border-radius: 6px !important;
            padding: 8px 22px !important;
            background: transparent !important;
            border: none !important;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: {T('accent')} !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }}

        div[data-testid="stFileUploader"] {{
            background-color: {T('bg2')} !important;
            border: 2px dashed {T('border')} !important;
            border-radius: 10px !important;
            padding: 0.25rem !important;
        }}

        div[data-testid="stFileUploader"]:hover {{
            border-color: {T('accent')} !important;
        }}

        div[data-testid="stFileUploaderDropzone"] {{
            background: transparent !important;
            padding: 1rem !important;
        }}

        div[data-testid="stFileUploaderDropzoneInstructions"] {{
            text-align: center !important;
        }}

        div[data-testid="stFileUploaderDropzoneInstructions"] > div {{
            display: none !important;
        }}

        div[data-testid="stFileUploaderDropzoneInstructions"]::before {{
            content: "Drop image here";
            display: block;
            font-size: 0.95rem;
            line-height: 1.4;
            color: {T('text2')};
            font-family: 'Inter', sans-serif !important;
            margin-bottom: 0.25rem;
        }}

        div[data-testid="stFileUploaderDropzoneInstructions"] small {{
            display: none !important;
        }}

        div[data-testid="stFileUploaderDropzone"] button {{
            margin-top: 0.35rem !important;
        }}

        [data-testid="stIconMaterial"] {{
            font-family: 'Material Symbols Rounded' !important;
            font-style: normal !important;
            font-weight: normal !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            font-feature-settings: 'liga' !important;
            -webkit-font-feature-settings: 'liga' !important;
            -webkit-font-smoothing: antialiased !important;
        }}

        .stButton > button {{
            background: transparent !important;
            color: {T('text2')} !important;
            border: 1px solid {T('border')} !important;
            border-radius: 6px !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            padding: 0.35rem 0.9rem !important;
            transition: all 0.15s !important;
        }}

        .stButton > button:hover {{
            border-color: {T('accent')} !important;
            color: {T('accent')} !important;
        }}

        [data-testid="stImage"] img {{
            border-radius: 8px !important;
            border: 1px solid {T('border')} !important;
        }}

        .ds-logo {{
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.6rem;
            font-weight: 700;
            color: {T('text1')};
            letter-spacing: 3px;
            text-transform: uppercase;
            line-height: 1;
        }}

        .ds-logo-accent {{
            color: {T('accent')};
        }}

        .ds-tagline {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: {T('text3')};
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-top: 5px;
        }}

        .ds-online {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: {T('text3')};
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 6px;
            margin-top: 24px;
        }}

        .ds-dot {{
            width: 7px;
            height: 7px;
            background: #43a047;
            border-radius: 50%;
            display: inline-block;
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.25; }}
        }}

        .ds-sh {{
            font-family: 'Inter', sans-serif;
            font-size: 0.68rem;
            font-weight: 700;
            color: {T('accent')};
            text-transform: uppercase;
            letter-spacing: 0.12em;
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 0 0 1rem 0;
        }}

        .ds-sh::after {{
            content: '';
            flex: 1;
            height: 1px;
            background: {T('border')};
        }}

        .ds-hint {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: {T('text3')};
            margin: 6px 0 0 2px;
        }}

        .ds-card {{
            background: {T('bg2')};
            border: 1px solid {T('border')};
            border-radius: 10px;
            padding: 18px;
            height: 100%;
        }}

        .ds-rcard {{
            background: {T('card')};
            border: 1px solid {T('border')};
            border-radius: 10px;
            padding: 24px 26px;
            position: relative;
            overflow: hidden;
            margin-bottom: 14px;
        }}

        .ds-rcard::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--cc);
        }}

        .ds-rover {{
            font-size: 0.65rem;
            font-weight: 700;
            color: {T('text3')};
            text-transform: uppercase;
            letter-spacing: 0.13em;
            margin-bottom: 6px;
        }}

        .ds-rclass {{
            font-family: 'Rajdhani', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--cc);
            text-transform: uppercase;
            letter-spacing: 3px;
            line-height: 1;
        }}

        .ds-rconf {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            color: {T('text2')};
            margin-top: 8px;
        }}

        .ds-badge {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.62rem;
            letter-spacing: 0.1em;
            padding: 3px 11px;
            border-radius: 20px;
            margin-top: 10px;
        }}

        .ds-action {{
            background: {T('bg3')};
            border-left: 3px solid var(--cc);
            border-radius: 0 6px 6px 0;
            padding: 12px 16px;
            font-size: 0.83rem;
            color: {T('text2')};
            line-height: 1.55;
            margin-bottom: 18px;
        }}

        .ds-action strong {{
            color: {T('text1')};
            font-weight: 600;
        }}

        .ds-brow {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 9px 0;
        }}

        .ds-blabel {{
            font-size: 0.8rem;
            font-weight: 500;
            color: {T('text3')};
            width: 82px;
            flex-shrink: 0;
            text-transform: capitalize;
        }}

        .ds-blabel.on {{
            color: {T('text1')};
            font-weight: 600;
        }}

        .ds-btrack {{
            flex: 1;
            background: {T('bar_bg')};
            border-radius: 3px;
            height: 7px;
            overflow: hidden;
        }}

        .ds-bfill {{
            height: 100%;
            border-radius: 3px;
        }}

        .ds-bpct {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            color: {T('text3')};
            width: 46px;
            text-align: right;
            flex-shrink: 0;
        }}

        .ds-bpct.on {{
            color: {T('text1')};
        }}

        .ds-callout {{
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 0.83rem;
            margin-top: 14px;
            line-height: 1.5;
        }}

        .ds-callout strong {{
            font-weight: 600;
        }}

        .ds-callout.good {{
            background: {T('success_bg')};
            color: {T('success_text')};
            border: 1px solid {T('success_text')}44;
        }}

        .ds-callout.warn {{
            background: {T('warn_bg')};
            color: {T('warn_text')};
            border: 1px solid {T('warn_text')}44;
        }}

        .ds-callout.info {{
            background: {T('info_bg')};
            color: {T('info_text')};
            border: 1px solid {T('info_text')}44;
        }}

        .ds-defect-title {{
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}

        .ds-about-title {{
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            font-weight: 700;
            color: {T('accent')};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 14px;
        }}

        .ds-about-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid {T('border')};
            gap: 10px;
        }}

        .ds-about-row:last-child {{
            border-bottom: none;
        }}

        .ds-about-key {{
            color: {T('text3')};
            font-size: 0.83rem;
        }}

        .ds-about-val {{
            color: {T('text1')};
            font-size: 0.83rem;
            font-weight: 500;
            text-align: right;
        }}

        .ds-bench-row {{
            display: grid;
            grid-template-columns: 1.1fr 1fr 1fr;
            gap: 12px;
            padding: 8px 0;
            border-bottom: 1px solid {T('border')};
            align-items: center;
        }}

        .ds-bench-row:last-child {{
            border-bottom: none;
        }}

        .ds-bench-row.head {{
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {T('text3')};
            padding-top: 0;
            padding-bottom: 10px;
        }}

        .ds-bench-metric {{
            color: {T('text3')};
            font-size: 0.83rem;
        }}

        .ds-bench-teacher {{
            color: {T('text2')};
            font-size: 0.83rem;
        }}

        .ds-bench-student {{
            color: {T('accent')};
            font-size: 0.83rem;
            font-weight: 600;
        }}

        .ds-footer {{
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.62rem;
            color: {T('text3')};
            letter-spacing: 0.04em;
            padding: 1rem 0 0.5rem 0;
        }}

        hr {{
            border-color: {T('border')} !important;
            margin: 2rem 0 !important;
        }}

        ::-webkit-scrollbar {{
            width: 5px;
        }}

        ::-webkit-scrollbar-track {{
            background: {T('bg')};
        }}

        ::-webkit-scrollbar-thumb {{
            background: {T('border')};
            border-radius: 3px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MODEL
# =============================================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model() -> nn.Module:
    if not os.path.isfile(WEIGHTS_PATH):
        st.error(f"Weights not found: `{WEIGHTS_PATH}`")
        st.stop()

    backbone = timm.create_model(
        STUDENT_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    backbone.load_state_dict(
        torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    )
    backbone.eval()

    return torch.ao.quantization.quantize_dynamic(
        backbone,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
    )


@st.cache_resource(show_spinner=False)
def get_transform() -> v2.Compose:
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


def predict(model: nn.Module, image: Image.Image) -> dict:
    tensor = get_transform()(image).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze().numpy()
    return {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}


# =============================================================================
# UI HELPERS
# =============================================================================
def section_header(title: str) -> None:
    st.markdown(f'<div class="ds-sh">{title}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, highlight: bool = False) -> None:
    value_color = T("accent") if highlight else T("text1")
    st.markdown(
        f"""
        <div class="ds-card" style="text-align:center;">
            <div style="font-family:'JetBrains Mono', monospace;font-size:1.45rem;font-weight:500;line-height:1;margin-bottom:7px;color:{value_color};">
                {value}
            </div>
            <div style="font-size:0.68rem;font-weight:600;color:{T('text3')};text-transform:uppercase;letter-spacing:0.09em;">
                {label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# NAV
# =============================================================================
def render_nav() -> None:
    col_logo, col_right = st.columns([3, 1])

    with col_logo:
        st.markdown(
            f"""
            <div style="padding: 1.8rem 0 1.4rem 0;">
                <div class="ds-logo">
                    DEFECT<span class="ds-logo-accent">SCAN</span>
                    <span style="color:{T('text3')};font-size:1rem;font-weight:400;letter-spacing:1px;"> AI</span>
                </div>
                <div class="ds-tagline">
                    3D Print Quality Control · Edge Inference · Knowledge Distillation
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            f"""
            <div class="ds-online">
                <span class="ds-dot"></span>
                <span style="color:{T('text3')};">MODEL ONLINE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin:0 0 2rem 0;'>", unsafe_allow_html=True)


# =============================================================================
# METRICS
# =============================================================================
def render_metrics() -> None:
    section_header("Performance Metrics")
    cols = st.columns(6)

    with cols[0]:
        metric_card("Test Accuracy", "99.63%", highlight=True)
    with cols[1]:
        metric_card("Macro F1", "0.9922", highlight=True)
    with cols[2]:
        metric_card("Model Size", "2.91 MB")
    with cols[3]:
        metric_card("Edge Speed", "152 FPS")
    with cols[4]:
        metric_card("vs Teacher", "1.93×")
    with cols[5]:
        metric_card("Parameters", "2.5M")


# =============================================================================
# VIDEO HELPERS
# =============================================================================
_VIDEO_SIZE_LIMIT = 150 * 1024 * 1024  # 150 MB


def extract_frames(video_url: str, n_frames: int = 8) -> list:
    """Download a video from *video_url* and return *n_frames* evenly-spaced
    PIL Images sampled across its full duration."""
    tmp_path = None
    try:
        resp = requests.get(
            video_url,
            stream=True,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()

        length = resp.headers.get("Content-Length")
        if length and int(length) > _VIDEO_SIZE_LIMIT:
            raise ValueError("Video is too large (> 150 MB). Try a shorter clip.")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded > _VIDEO_SIZE_LIMIT:
                    raise ValueError("Video is too large (> 150 MB). Try a shorter clip.")

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video — make sure the URL points to a direct MP4/WebM file.")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError("Video has no readable frames.")

        frames: list = []
        for i in range(n_frames):
            idx = int(i * (total - 1) / max(n_frames - 1, 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

        if not frames:
            raise ValueError("Could not extract any frames from the video.")
        return frames

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def render_video_results(model: nn.Module, frames: list) -> None:
    """Run inference on each frame and display a grid + overall verdict."""
    results = []
    for img in frames:
        probs = predict(model, img)
        winner = max(probs, key=probs.get)
        results.append((img, winner, probs[winner], probs))

    sev_rank = {"CRITICAL": 3, "MODERATE": 2, "MINOR": 1}
    worst = max(results, key=lambda r: (sev_rank.get(CLASS_META[r[1]]["severity"], 0), r[2]))
    worst_cls, worst_conf = worst[1], worst[2]
    worst_meta = CLASS_META[worst_cls]

    badge_styles = {
        "CRITICAL": "background:#e5393522;color:#ef9a9a;border:1px solid #e5393566;",
        "MODERATE": "background:#ff704322;color:#ffcc80;border:1px solid #ff704366;",
        "MINOR":    "background:#e5737322;color:#ffcdd2;border:1px solid #e5737366;",
    }

    callout_cls = {"spaghetti": "warn", "stringing": "info", "zits": "good"}.get(worst_cls, "warn")
    badge_inline = badge_styles[worst_meta["severity"]] + "padding:2px 8px;border-radius:10px;font-size:0.75rem;"
    st.markdown(
        f'<div class="ds-callout {callout_cls}" style="margin-bottom:1.2rem;">'
        f'<strong>Overall verdict:</strong> '
        f'{worst_meta["icon"]} <strong>{worst_cls.upper()}</strong> detected at '
        f'{worst_conf * 100:.1f}% confidence across {len(frames)} sampled frames. '
        f'<span style="{badge_inline}">{worst_meta["severity"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cols_per_row = 4
    for row_start in range(0, len(results), cols_per_row):
        row = results[row_start : row_start + cols_per_row]
        cols = st.columns(len(row))
        for col, (img, cls, conf, _probs) in zip(cols, row):
            meta = CLASS_META[cls]
            with col:
                st.image(img, use_container_width=True)
                st.markdown(
                    f'<div style="text-align:center;margin-top:4px;">'
                    f'<span style="font-size:0.78rem;font-weight:600;color:{meta["hex"]};">'
                    f'{meta["icon"]} {cls}</span><br>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
                    f'color:{T("text3")};">{conf * 100:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# =============================================================================
# CLASSIFIER
# =============================================================================
def render_classifier(model: nn.Module) -> None:
    section_header("Defect Classifier")

    tab_upload, tab_video = st.tabs(["📁 Upload Image", "🎬 Video Analysis"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )

        st.markdown(
            '<p class="ds-hint">Supported formats: JPG, PNG, WEBP, BMP</p>',
            unsafe_allow_html=True,
        )

        if uploaded is not None:
            image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
            st.markdown("<hr>", unsafe_allow_html=True)
            with st.spinner("Running inference..."):
                probs = predict(model, image)
            render_results(probs, image)

    with tab_video:
        st.markdown(
            '<p class="ds-hint" style="margin-bottom:0.6rem;">Paste a direct link to an MP4 or WebM video '
            '(GitHub raw, public CDN, etc.) — the model will scan frames across the full clip.</p>',
            unsafe_allow_html=True,
        )

        video_url = st.text_input(
            "Video URL",
            placeholder="https://example.com/print_defect.mp4",
            label_visibility="collapsed",
        )

        if video_url:
            col_vid, col_ctrl = st.columns([2, 1], gap="large")

            with col_vid:
                try:
                    st.video(video_url)
                except Exception:
                    st.markdown(
                        '<p class="ds-hint">⚠ Preview unavailable — the video will still be analysed.</p>',
                        unsafe_allow_html=True,
                    )

            with col_ctrl:
                n_frames = st.slider("Frames to sample", min_value=4, max_value=16, value=8, step=2)
                st.markdown(
                    '<p class="ds-hint">More frames = slower but more thorough scan.</p>',
                    unsafe_allow_html=True,
                )
                run_scan = st.button("🔍 Scan for Defects", use_container_width=True)

            if run_scan:
                st.markdown("<hr>", unsafe_allow_html=True)
                with st.spinner(f"Downloading video and scanning {n_frames} frames…"):
                    try:
                        frames = extract_frames(video_url, n_frames)
                        render_video_results(model, frames)
                    except Exception as exc:
                        st.error(f"Could not process video: {exc}")


# =============================================================================
# RESULTS
# =============================================================================
def render_results(probs: dict, image: Image.Image) -> None:
    winner = max(probs, key=probs.get)
    wconf = probs[winner]
    meta = CLASS_META[winner]
    cc = meta["hex"]
    sev = meta["severity"]

    badge_styles = {
        "CRITICAL": "background:#e5393522;color:#ef9a9a;border:1px solid #e5393566;",
        "MODERATE": "background:#ff704322;color:#ffcc80;border:1px solid #ff704366;",
        "MINOR": "background:#e5737322;color:#ffcdd2;border:1px solid #e5737366;",
    }

    col_img, col_res = st.columns([1, 1.25], gap="large")

    with col_img:
        section_header("Input Image")
        st.image(image, use_container_width=True)
        w, h = image.size
        st.markdown(
            f"""
            <p style="font-family:'JetBrains Mono', monospace;font-size:0.7rem;color:{T('text3')};margin-top:5px;">
                {w} × {h} px
            </p>
            """,
            unsafe_allow_html=True,
        )

    with col_res:
        section_header("Result")

        st.markdown(
            f"""
            <div class="ds-rcard" style="--cc:{cc};">
                <div class="ds-rover">Detected Defect</div>
                <div class="ds-rclass">{winner}</div>
                <div class="ds-rconf">{wconf * 100:.2f}% confidence</div>
                <div class="ds-badge" style="{badge_styles[sev]}">{sev}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="ds-action" style="--cc:{cc};">
                <strong>Action:</strong> {meta['action']}<br>
                {meta['description']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        section_header("Confidence")

        for cls in CLASS_NAMES:
            p = probs[cls]
            active = cls == winner
            fill_color = CLASS_META[cls]["hex"] if active else T("border")
            glow = f"box-shadow:0 0 6px {fill_color}66;" if active else ""

            st.markdown(
                f"""
                <div class="ds-brow">
                    <div class="ds-blabel {'on' if active else ''}">{cls}</div>
                    <div class="ds-btrack">
                        <div class="ds-bfill" style="width:{p * 100:.1f}%;background:{fill_color};{glow}"></div>
                    </div>
                    <div class="ds-bpct {'on' if active else ''}">{p * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        sorted_probs = sorted(probs.values(), reverse=True)
        gap = sorted_probs[0] - sorted_probs[1]

        if gap >= 0.80:
            box_class = "good"
            msg = f"<strong>High certainty</strong> {gap * 100:.1f}% margin over second class."
        elif gap >= 0.40:
            box_class = "info"
            msg = f"<strong>Clear result</strong> {gap * 100:.1f}% margin over second class."
        else:
            box_class = "warn"
            msg = f"<strong>Low confidence</strong> only {gap * 100:.1f}% margin. Try a clearer image."

        st.markdown(
            f'<div class="ds-callout {box_class}">{msg}</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# DEFECT REFERENCE
# =============================================================================
def render_defect_reference() -> None:
    section_header("Defect Reference")
    cols = st.columns(3)

    for i, cls in enumerate(CLASS_NAMES):
        meta = CLASS_META[cls]

        with cols[i]:
            st.markdown(
                f"""
                <div class="ds-card" style="border-top:3px solid {meta['hex']}; min-height:220px;">
                    <div class="ds-defect-title" style="color:{meta['hex']};">
                        {meta['icon']} {cls}
                    </div>
                    <div style="font-family:'JetBrains Mono', monospace;font-size:0.65rem;color:{T('text3')};letter-spacing:0.1em;margin-bottom:10px;">
                        {meta['severity']}
                    </div>
                    <div style="font-size:0.82rem;color:{T('text2')};line-height:1.55;margin-bottom:10px;">
                        {meta['description']}
                    </div>
                    <div style="font-size:0.78rem;color:{T('text2')};">
                        <strong style="color:{T('text1')};">Action:</strong> {meta['action']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# ABOUT
# =============================================================================
def render_about() -> None:
    section_header("About This Model")

    architecture_rows = [
        ("Teacher", "MobileNetV4-Medium"),
        ("Student", "MobileNetV4-Small"),
        ("Parameters", "2.5M"),
        ("Distillation", "KL + Attention Transfer"),
        ("Quantization", "Dynamic INT8"),
        ("Training epochs", "150"),
        ("Dataset", "8,851 images · 3 classes"),
    ]

    architecture_html = "".join(
        f'<div class="ds-about-row"><span class="ds-about-key">{label}</span>'
        f'<span class="ds-about-val">{value}</span></div>'
        for label, value in architecture_rows
    )

    benchmark_html = (
        '<div class="ds-bench-row head"><span>Metric</span>'
        '<span>Teacher FP32</span>'
        '<span style="color:#e53935;">Student INT8</span></div>'
        '<div class="ds-bench-row"><span class="ds-bench-metric">Avg latency</span>'
        '<span class="ds-bench-teacher">12.68 ms</span>'
        '<span class="ds-bench-student">6.56 ms</span></div>'
        '<div class="ds-bench-row"><span class="ds-bench-metric">P95 latency</span>'
        '<span class="ds-bench-teacher">15.47 ms</span>'
        '<span class="ds-bench-student">7.41 ms</span></div>'
        '<div class="ds-bench-row"><span class="ds-bench-metric">Jitter (\u03c3)</span>'
        '<span class="ds-bench-teacher">1.32 ms</span>'
        '<span class="ds-bench-student">0.44 ms \u2713</span></div>'
        '<div class="ds-bench-row"><span class="ds-bench-metric">FPS</span>'
        '<span class="ds-bench-teacher">78.8</span>'
        '<span class="ds-bench-student">152.4</span></div>'
        '<div class="ds-bench-row"><span class="ds-bench-metric">Speedup</span>'
        '<span class="ds-bench-teacher">\u2014</span>'
        '<span class="ds-bench-student">1.93\u00d7</span></div>'
    )

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown(
            f'<div class="ds-card"><div class="ds-about-title">Architecture</div>{architecture_html}</div>',
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            f'<div class="ds-card"><div class="ds-about-title">Edge Benchmark (4-thread CPU)</div>{benchmark_html}</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    inject_css()
    model = load_model()

    render_nav()
    render_metrics()

    st.markdown("<hr>", unsafe_allow_html=True)
    render_classifier(model)

    st.markdown("<hr>", unsafe_allow_html=True)
    render_defect_reference()

    st.markdown("<hr>", unsafe_allow_html=True)
    render_about()

    st.markdown(
        """
        <div class="ds-footer">
            DefectScan AI · MobileNetV4-Small (2.5M)
            · Distilled from MobileNetV4-Medium (8.7M)
            · PyTorch 2.5 · timm · Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()