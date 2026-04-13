"""
Intelligent Multi-Stage Chest X-Ray Analysis System
Streamlit UI — v2.0
Author: Chenduluru Siva | 7151CEM | Coventry University
"""

import os
import sys
import numpy as np
import torch
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import DISEASE_LABELS, SAVED_MODELS_DIR, BACKBONE, CLASSIFICATION_THRESHOLD
from src.preprocessing.quality_assessment import ImageQualityAssessor
from src.preprocessing.dataset import get_eval_transform
from src.models.densenet_model import build_model, load_checkpoint
from src.explainability.gradcam import GradCAM, denormalise
from src.reporting.report_generator import ReportGenerator

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChestAI — Intelligent X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00b4d8, #0077b6, #48cae4, #00b4d8);
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.header-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e6edf3;
    margin: 0;
    letter-spacing: -0.5px;
}
.header-title span { color: #00b4d8; }
.header-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #8b949e;
    margin-top: 0.4rem;
    letter-spacing: 1px;
}

/* Stage cards */
.stage-row {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 1.5rem;
}
.stage-card {
    flex: 1;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    text-align: center;
    position: relative;
    transition: border-color 0.2s;
}
.stage-card.active  { border-color: #00b4d8; }
.stage-card.done    { border-color: #3fb950; }
.stage-card.pending { border-color: #30363d; opacity: 0.5; }
.stage-card .s-icon { font-size: 1.4rem; display: block; margin-bottom: 0.3rem; }
.stage-card .s-num  {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #8b949e;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.stage-card .s-name {
    font-size: 0.75rem;
    font-weight: 600;
    color: #e6edf3;
    margin-top: 0.15rem;
}
.stage-card.active .s-name  { color: #00b4d8; }
.stage-card.done   .s-name  { color: #3fb950; }

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem;
    margin-bottom: 1rem;
}
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
.metric-card .m-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #8b949e;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-card .m-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .m-sub {
    font-size: 0.65rem;
    color: #8b949e;
    margin-top: 0.15rem;
}
.metric-card.green { border-color: #3fb950; }
.metric-card.green .m-value { color: #3fb950; }
.metric-card.red   { border-color: #f85149; }
.metric-card.red   .m-value { color: #f85149; }
.metric-card.blue  { border-color: #00b4d8; }
.metric-card.blue  .m-value { color: #00b4d8; }
.metric-card.amber { border-color: #d29922; }
.metric-card.amber .m-value { color: #d29922; }

/* Finding badges */
.finding-urgent   { background:#3d0f0f; border:1px solid #f85149; color:#ffa198; border-radius:4px; padding:2px 8px; font-size:0.7rem; font-weight:700; font-family:'JetBrains Mono',monospace; }
.finding-priority { background:#2d2000; border:1px solid #d29922; color:#e3b341; border-radius:4px; padding:2px 8px; font-size:0.7rem; font-weight:700; font-family:'JetBrains Mono',monospace; }
.finding-routine  { background:#0d2818; border:1px solid #3fb950; color:#56d364; border-radius:4px; padding:2px 8px; font-size:0.7rem; font-weight:700; font-family:'JetBrains Mono',monospace; }

/* Section headers */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #21262d;
}

/* Report box */
.report-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #c9d1d9;
    white-space: pre-wrap;
    line-height: 1.6;
    max-height: 400px;
    overflow-y: auto;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

/* Buttons */
.stButton > button {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    font-size: 0.8rem;
    padding: 0.5rem 1.2rem;
    transition: background 0.2s;
}
.stButton > button:hover { background: #388bfd; }

/* Upload area */
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 8px;
    padding: 1rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border-radius: 6px;
    font-family: 'Sora', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #e6edf3 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #161b22; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

/* Info/warning boxes */
.info-box {
    background: #0c2d6b;
    border: 1px solid #1f6feb;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #79c0ff;
    margin-bottom: 1rem;
}
.warn-box {
    background: #2d1c00;
    border: 1px solid #d29922;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #e3b341;
    margin-bottom: 1rem;
}
.success-box {
    background: #0d2818;
    border: 1px solid #3fb950;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #56d364;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading DenseNet-121 model …")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(BACKBONE).to(device)
    ckpt   = os.path.join(SAVED_MODELS_DIR, f"best_{BACKBONE}.pt")
    if os.path.exists(ckpt):
        model  = load_checkpoint(model, ckpt, device)
        status = "loaded"
        epoch  = torch.load(ckpt, map_location=device).get("epoch", "?")
        auroc  = torch.load(ckpt, map_location=device).get("val_auroc", 0)
    else:
        status = "demo"
        epoch  = 0
        auroc  = 0.0
    return model, device, status, epoch, auroc

@st.cache_resource
def get_assessor():  return ImageQualityAssessor()

@st.cache_resource
def get_reporter():  return ReportGenerator()


# ── Helpers ───────────────────────────────────────────────────────────────────
def predict(model, device, pil_image):
    transform = get_eval_transform()
    tensor    = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model.predict_proba(tensor).cpu().numpy()[0]
    return probs, tensor


def make_gradcam_figure(model, tensor, probs, device, top_k=4):
    """Generate Grad-CAM with dark theme matplotlib figure."""
    sorted_idx = np.argsort(probs)[::-1][:top_k]
    gcam       = GradCAM(model)
    orig_np    = denormalise(tensor.squeeze(0))

    fig, axes = plt.subplots(1, top_k + 1, figsize=(4 * (top_k + 1), 4))
    fig.patch.set_facecolor('#0d1117')

    axes[0].imshow(orig_np, cmap='gray')
    axes[0].set_title('Original X-Ray', color='#e6edf3', fontsize=9,
                      fontfamily='monospace', pad=8)
    axes[0].axis('off')
    for spine in axes[0].spines.values():
        spine.set_edgecolor('#30363d')

    colors = ['#00b4d8', '#3fb950', '#d29922', '#f85149', '#a371f7', '#ff7b72']
    for col, idx in enumerate(sorted_idx, start=1):
        try:
            hm = gcam.generate(tensor.to(device), int(idx))
            ov = gcam.overlay(orig_np, hm)
            axes[col].imshow(ov)
        except Exception:
            axes[col].imshow(orig_np, cmap='gray')
        prob  = probs[idx]
        label = DISEASE_LABELS[idx]
        axes[col].set_title(
            f'{label}\np = {prob:.3f}',
            color=colors[col % len(colors)],
            fontsize=8.5, fontfamily='monospace', pad=8
        )
        axes[col].axis('off')
        for spine in axes[col].spines.values():
            spine.set_edgecolor(colors[col % len(colors)])
            spine.set_linewidth(1.5)

    plt.tight_layout(pad=0.5)
    return fig


def make_attention_figure(model, tensor, probs, device):
    """
    Multi-Head Attention Map visualisation.
    Shows raw DenseNet feature activations as attention-style heatmap.
    """
    model.eval()
    activations = {}

    def hook_fn(module, inp, out):
        activations['feat'] = out.detach()

    # Register hook on denseblock4
    handle = model.features.denseblock4.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(tensor.to(device))
    handle.remove()

    feat = activations['feat'][0]          # (C, H, W)
    orig_np = denormalise(tensor.squeeze(0))

    # Compute channel-wise attention (mean + max combination)
    mean_attn = feat.mean(dim=0).cpu().numpy()
    max_attn  = feat.max(dim=0).values.cpu().numpy()

    # Normalise each
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    mean_map = cv2.resize(norm(mean_attn), (224, 224))
    max_map  = cv2.resize(norm(max_attn),  (224, 224))
    combined = 0.5 * mean_map + 0.5 * max_map

    # Top disease channel maps
    top_idx = np.argsort(probs)[::-1][:3]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('DenseNet-121 Feature Attention Maps', color='#e6edf3',
                 fontsize=11, fontfamily='monospace', y=1.01)

    titles  = ['Original', 'Mean Activation', 'Max Activation', 'Combined']
    maps    = [None, mean_map, max_map, combined]
    cmaps   = ['gray', 'plasma', 'inferno', 'viridis']

    for col, (title, m, cmap) in enumerate(zip(titles, maps, cmaps)):
        ax = axes[0][col]
        if m is None:
            ax.imshow(orig_np, cmap='gray')
        else:
            ax.imshow(orig_np, cmap='gray', alpha=0.4)
            ax.imshow(m, cmap=cmap, alpha=0.7)
        ax.set_title(title, color='#8b949e', fontsize=8.5, fontfamily='monospace')
        ax.axis('off')

    # Per-disease top channel attention
    disease_colors = ['#00b4d8', '#3fb950', '#d29922']
    for col2, (idx, dc) in enumerate(zip(top_idx, disease_colors)):
        ax = axes[1][col2]
        # Get feature channels most correlated with this disease
        label = DISEASE_LABELS[idx]
        prob  = probs[idx]
        # Use top-N channels by activation magnitude
        ch_means  = feat.mean(dim=(1, 2)).cpu().numpy()
        top_chs   = np.argsort(ch_means)[-20:]
        ch_map    = feat[top_chs].mean(dim=0).cpu().numpy()
        ch_map    = cv2.resize(norm(ch_map), (224, 224))
        ax.imshow(orig_np, cmap='gray', alpha=0.35)
        im = ax.imshow(ch_map, cmap='hot', alpha=0.75)
        ax.set_title(f'{label}\np={prob:.3f}',
                     color=dc, fontsize=8.5, fontfamily='monospace')
        ax.axis('off')

    axes[1][3].axis('off')
    axes[1][3].set_facecolor('#0d1117')

    plt.tight_layout(pad=0.8)
    return fig


def make_probability_radar(probs):
    """Radar/polar chart of all 14 disease probabilities."""
    categories = DISEASE_LABELS
    values     = probs.tolist()
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=cats_closed,
        fill='toself',
        fillcolor='rgba(0, 180, 216, 0.15)',
        line=dict(color='#00b4d8', width=2),
        name='Disease Probability'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[0.5] * (len(categories) + 1),
        theta=cats_closed,
        mode='lines',
        line=dict(color='rgba(248,81,73,0.4)', width=1, dash='dash'),
        name='Threshold (0.5)'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#161b22',
            radialaxis=dict(
                visible=True, range=[0, 1],
                color='#8b949e', gridcolor='#30363d',
                tickfont=dict(size=8, color='#8b949e', family='JetBrains Mono')
            ),
            angularaxis=dict(
                color='#8b949e', gridcolor='#30363d',
                tickfont=dict(size=7.5, color='#c9d1d9', family='JetBrains Mono')
            )
        ),
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        legend=dict(
            font=dict(color='#8b949e', size=10, family='JetBrains Mono'),
            bgcolor='#161b22', bordercolor='#30363d'
        ),
        margin=dict(t=30, b=30, l=60, r=60),
        height=380
    )
    return fig


def make_probability_bars(probs, threshold):
    """Horizontal bar chart for all 14 probabilities."""
    df = pd.DataFrame({'Disease': DISEASE_LABELS, 'Probability': probs})
    df = df.sort_values('Probability', ascending=True)

    colors = []
    for p in df['Probability']:
        if p >= 0.7:   colors.append('#f85149')
        elif p >= 0.5: colors.append('#d29922')
        else:           colors.append('#30363d')

    fig = go.Figure(go.Bar(
        x=df['Probability'], y=df['Disease'],
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f'{p:.3f}' for p in df['Probability']],
        textposition='outside',
        textfont=dict(size=9, color='#8b949e', family='JetBrains Mono')
    ))
    fig.add_vline(
        x=threshold, line_dash='dash',
        line_color='rgba(0,180,216,0.6)', line_width=1.5,
        annotation_text=f'Threshold {threshold:.2f}',
        annotation_font=dict(color='#00b4d8', size=9, family='JetBrains Mono')
    )
    fig.update_layout(
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9', family='JetBrains Mono'),
        xaxis=dict(range=[0, 1.1], gridcolor='#21262d', color='#8b949e',
                   tickfont=dict(size=8)),
        yaxis=dict(color='#c9d1d9', tickfont=dict(size=8.5)),
        margin=dict(t=10, b=10, l=10, r=60),
        height=400, showlegend=False
    )
    return fig


def make_qa_gauge(score):
    """Gauge chart for quality assessment score."""
    color = '#3fb950' if score >= 70 else '#d29922' if score >= 50 else '#f85149'
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number=dict(
            suffix='/100', font=dict(size=28, color=color,
            family='JetBrains Mono')
        ),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor='#8b949e',
                      tickfont=dict(color='#8b949e', size=9)),
            bar=dict(color=color, thickness=0.25),
            bgcolor='#161b22',
            bordercolor='#30363d',
            steps=[
                dict(range=[0,  50], color='rgba(248,81,73,0.1)'),
                dict(range=[50, 70], color='rgba(210,153,34,0.1)'),
                dict(range=[70,100], color='rgba(63,185,80,0.1)')
            ],
            threshold=dict(
                line=dict(color='#00b4d8', width=2),
                thickness=0.75, value=70
            )
        )
    ))
    fig.update_layout(
        paper_bgcolor='#0d1117', font=dict(color='#c9d1d9'),
        margin=dict(t=20, b=20, l=20, r=20), height=200
    )
    return fig


URGENCY_MAP = {
    "Pneumothorax": "URGENT",  "Pneumonia":  "PRIORITY",
    "Effusion":     "PRIORITY", "Edema":     "PRIORITY",
    "Mass":         "PRIORITY", "Consolidation": "PRIORITY",
    "Cardiomegaly": "ROUTINE",  "Atelectasis":   "ROUTINE",
    "Nodule":       "ROUTINE",  "Infiltration":  "ROUTINE",
    "Emphysema":    "ROUTINE",  "Fibrosis":      "ROUTINE",
    "Pleural_Thickening": "ROUTINE", "Hernia":   "ROUTINE"
}


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

# Load resources
model, device, model_status, ckpt_epoch, ckpt_auroc = load_model()
assessor = get_assessor()
reporter = get_reporter()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
                    letter-spacing:3px;color:#8b949e;text-transform:uppercase'>
            ChestAI System
        </div>
        <div style='font-size:1.1rem;font-weight:700;color:#e6edf3;margin-top:4px'>
            Control Panel
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model status
    if model_status == "loaded":
        st.markdown(f"""
        <div class='success-box'>
        ✅ Model loaded<br>
        <span style='font-family:JetBrains Mono,monospace;font-size:0.65rem'>
        Epoch {ckpt_epoch} · AUROC {ckpt_auroc:.4f}
        </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='warn-box'>
        ⚠️ Demo mode — no checkpoint<br>
        <span style='font-family:JetBrains Mono,monospace;font-size:0.65rem'>
        Run training first
        </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:#161b22;border:1px solid #30363d;border-radius:8px;
                padding:0.7rem 0.9rem;margin-bottom:1rem;font-family:JetBrains Mono,
                monospace;font-size:0.68rem;color:#8b949e'>
    Device: <span style='color:#e6edf3'>
    {"🟢 GPU" if device.type=="cuda" else "🟡 CPU"}</span><br>
    Backbone: <span style='color:#00b4d8'>DenseNet-121</span><br>
    Classes: <span style='color:#e6edf3'>14 diseases</span><br>
    Dataset: <span style='color:#e6edf3'>NIH ChestX-ray14</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Detection Settings**")
    threshold = st.slider(
        "Threshold", 0.1, 0.9,
        float(CLASSIFICATION_THRESHOLD), 0.05,
        help="Probability threshold for positive detection"
    )

    st.markdown("**🗺️ Visualisation**")
    show_gradcam   = st.checkbox("Grad-CAM Heatmaps",    value=True)
    show_attention = st.checkbox("Attention Feature Maps", value=True)
    show_radar     = st.checkbox("Probability Radar",     value=True)
    top_k          = st.slider("Top-K diseases to show", 1, 6, 4)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;
                color:#8b949e;line-height:1.8'>
    <b style='color:#c9d1d9'>Pipeline Stages</b><br>
    1 · Quality Assessment<br>
    2 · DenseNet Classification<br>
    3 · Grad-CAM Explainability<br>
    4 · Attention Feature Maps<br>
    5 · Structured Report
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <div class='header-title'>
        🫁 Chest<span>AI</span> — Intelligent X-Ray Analysis System
    </div>
    <div class='header-sub'>
        7151CEM · MSc Computing · Coventry University · Chenduluru Siva
        &nbsp;·&nbsp; DenseNet-121 · NIH ChestX-ray14 · 14 Thoracic Diseases
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Single Analysis",
    "📦 Batch Processing",
    "📊 Model Dashboard",
    "📖 About"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    uploaded = st.file_uploader(
        "Drop a chest X-ray here or click to browse (PNG / JPG)",
        type=["png", "jpg", "jpeg"],
        key="single_upload"
    )

    if not uploaded:
        # Welcome state
        c1, c2, c3, c4 = st.columns(4)
        for col, (icon, num, name) in zip(
            [c1, c2, c3, c4],
            [("🔍","01","Quality Check"),
             ("🧠","02","Classification"),
             ("🗺️","03","Explainability"),
             ("📄","04","Report")]
        ):
            col.markdown(f"""
            <div class='stage-card pending'>
                <span class='s-icon'>{icon}</span>
                <div class='s-num'>Stage {num}</div>
                <div class='s-name'>{name}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-box' style='margin-top:1.5rem;text-align:center'>
        ⬆️ Upload a frontal chest X-ray to begin the 4-stage analysis pipeline
        </div>
        """, unsafe_allow_html=True)

    else:
        pil_image = Image.open(uploaded).convert("RGB")

        # ── STAGE 1: Quality Assessment ───────────────────────────────────────
        st.markdown("<div class='section-header'>Stage 01 — Image Quality Assessment</div>",
                    unsafe_allow_html=True)

        img_gray = np.array(pil_image.convert("L"))
        qa       = assessor.assess(img_gray)

        col_gauge, col_metrics = st.columns([1, 2])

        with col_gauge:
            st.plotly_chart(make_qa_gauge(qa.overall_score),
                            use_container_width=True)

        with col_metrics:
            color_class = "green" if qa.is_acceptable else "red"
            st.markdown(f"""
            <div class='metric-grid' style='grid-template-columns:repeat(2,1fr)'>
                <div class='metric-card blue'>
                    <div class='m-label'>Brightness</div>
                    <div class='m-value'>{qa.brightness:.0f}</div>
                    <div class='m-sub'>Target: 30–220</div>
                </div>
                <div class='metric-card blue'>
                    <div class='m-label'>Contrast</div>
                    <div class='m-value'>{qa.contrast:.0f}</div>
                    <div class='m-sub'>Min: 20</div>
                </div>
                <div class='metric-card {"green" if qa.sharpness>=100 else "red"}'>
                    <div class='m-label'>Sharpness</div>
                    <div class='m-value'>{qa.sharpness:.0f}</div>
                    <div class='m-sub'>Min: 100 (Laplacian)</div>
                </div>
                <div class='metric-card {"green" if qa.black_ratio<=0.3 else "red"}'>
                    <div class='m-label'>Black Ratio</div>
                    <div class='m-value'>{qa.black_ratio:.1%}</div>
                    <div class='m-sub'>Max: 30%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if qa.is_acceptable:
                st.markdown(f"""
                <div class='success-box'>
                ✅ <b>ACCEPTED</b> — {qa.recommendation}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='warn-box'>
                ⚠️ <b>CAUTION</b> — {qa.recommendation}<br>
                {'<br>'.join(['• ' + i for i in qa.issues])}
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── STAGE 2: Classification ───────────────────────────────────────────
        st.markdown("<div class='section-header'>Stage 02 — Multi-Label Disease Classification (DenseNet-121)</div>",
                    unsafe_allow_html=True)

        with st.spinner("Running DenseNet-121 forward pass …"):
            probs, tensor = predict(model, device, pil_image)

        positive = [(DISEASE_LABELS[i], probs[i])
                    for i in range(14) if probs[i] >= threshold]
        positive.sort(key=lambda x: -x[1])

        col_img, col_findings, col_chart = st.columns([1, 1.2, 1.5])

        with col_img:
            st.image(pil_image, caption=uploaded.name, use_column_width=True)

        with col_findings:
            st.markdown("**Detected Findings**")
            if positive:
                for label, p in positive:
                    urg    = URGENCY_MAP.get(label, "ROUTINE")
                    badge  = f"finding-{urg.lower()}"
                    conf   = "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.5 else "LOW"
                    st.markdown(f"""
                    <div style='background:#161b22;border:1px solid #30363d;
                                border-radius:6px;padding:0.6rem 0.8rem;
                                margin-bottom:0.4rem'>
                        <div style='display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:0.3rem'>
                            <span style='font-weight:600;font-size:0.82rem;
                                         color:#e6edf3'>{label}</span>
                            <span class='{badge}'>{urg}</span>
                        </div>
                        <div style='display:flex;justify-content:space-between;
                                    align-items:center'>
                            <span style='font-family:JetBrains Mono,monospace;
                                         font-size:0.75rem;color:#00b4d8'>
                                p = {p:.4f}</span>
                            <span style='font-family:JetBrains Mono,monospace;
                                         font-size:0.65rem;color:#8b949e'>
                                {conf}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='success-box'>
                    ✅ No significant findings above threshold
                </div>""", unsafe_allow_html=True)

            # Summary stats
            st.markdown(f"""
            <div style='background:#161b22;border:1px solid #30363d;border-radius:6px;
                        padding:0.6rem 0.8rem;margin-top:0.5rem;
                        font-family:JetBrains Mono,monospace;font-size:0.7rem;
                        color:#8b949e'>
            Threshold: <span style='color:#00b4d8'>{threshold:.2f}</span> &nbsp;·&nbsp;
            Detected: <span style='color:#{"f85149" if positive else "3fb950"}'>
            {len(positive)}</span> / 14 &nbsp;·&nbsp;
            Max prob: <span style='color:#e6edf3'>{probs.max():.4f}</span>
            </div>
            """, unsafe_allow_html=True)

        with col_chart:
            st.plotly_chart(make_probability_bars(probs, threshold),
                            use_container_width=True)

        # All probabilities expander
        with st.expander("📊 Full probability table — all 14 conditions"):
            df_all = pd.DataFrame({
                "Disease":     DISEASE_LABELS,
                "Probability": [round(float(p), 4) for p in probs],
                "Detected":    ["✅ YES" if p >= threshold else "—" for p in probs],
                "Urgency":     [URGENCY_MAP.get(l, "ROUTINE") for l in DISEASE_LABELS],
                "Confidence":  ["HIGH" if p>=0.7 else "MEDIUM" if p>=0.5 else "LOW"
                                for p in probs]
            }).sort_values("Probability", ascending=False).reset_index(drop=True)
            st.dataframe(df_all, use_container_width=True)

        st.markdown("---")

        # ── Radar chart ───────────────────────────────────────────────────────
        if show_radar:
            st.markdown("<div class='section-header'>Probability Radar — All 14 Conditions</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(make_probability_radar(probs), use_container_width=True)
            st.markdown("---")

        # ── STAGE 3: Grad-CAM ─────────────────────────────────────────────────
        if show_gradcam:
            st.markdown("<div class='section-header'>Stage 03 — Grad-CAM Explainability (Selvaraju et al., 2017)</div>",
                        unsafe_allow_html=True)
            st.caption("Heatmaps show which image regions most influenced each disease prediction.")

            with st.spinner("Computing gradients and generating Grad-CAM …"):
                try:
                    fig_gcam = make_gradcam_figure(model, tensor, probs, device, top_k)
                    st.pyplot(fig_gcam, use_container_width=True)
                    plt.close(fig_gcam)
                except Exception as e:
                    st.warning(f"Grad-CAM error: {e}")
            st.markdown("---")

        # ── STAGE 4: Attention Feature Maps ───────────────────────────────────
        if show_attention:
            st.markdown("<div class='section-header'>Stage 04 — DenseNet-121 Attention Feature Maps</div>",
                        unsafe_allow_html=True)
            st.caption("Visualises internal feature activations from DenseBlock-4. Shows mean, max, and combined channel attention alongside per-disease activation patterns.")

            with st.spinner("Extracting DenseNet-121 feature maps …"):
                try:
                    fig_att = make_attention_figure(model, tensor, probs, device)
                    st.pyplot(fig_att, use_container_width=True)
                    plt.close(fig_att)
                except Exception as e:
                    st.warning(f"Attention map error: {e}")
            st.markdown("---")

        # ── STAGE 5: Structured Report ────────────────────────────────────────
        st.markdown("<div class='section-header'>Stage 05 — Automated Structured Report</div>",
                    unsafe_allow_html=True)

        report = reporter.generate(
            probs=probs, filename=uploaded.name,
            qa_score=qa.overall_score,
            qa_acceptable=qa.is_acceptable
        )

        st.markdown(f"<div class='report-box'>{report.full_text}</div>",
                    unsafe_allow_html=True)

        col_dl1, col_dl2, _ = st.columns([1, 1, 3])
        with col_dl1:
            st.download_button(
                "⬇️ Download Report (.txt)",
                data=report.full_text,
                file_name=f"{report.report_id}.txt",
                mime="text/plain"
            )
        with col_dl2:
            # Export CSV of predictions
            csv_data = pd.DataFrame({
                "Disease":     DISEASE_LABELS,
                "Probability": probs.tolist(),
                "Detected":    [p >= threshold for p in probs]
            }).to_csv(index=False)
            st.download_button(
                "⬇️ Download Predictions (.csv)",
                data=csv_data,
                file_name=f"predictions_{uploaded.name}.csv",
                mime="text/csv"
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Batch X-Ray Processing</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    Upload multiple chest X-rays to analyse them in sequence.
    Results are compiled into a downloadable summary table.
    </div>
    """, unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload multiple X-rays",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if batch_files:
        results    = []
        prog_bar   = st.progress(0)
        status_txt = st.empty()

        for i, f in enumerate(batch_files):
            status_txt.markdown(
                f"<div class='info-box'>Processing {f.name} "
                f"({i+1}/{len(batch_files)}) …</div>",
                unsafe_allow_html=True
            )
            pil  = Image.open(f).convert("RGB")
            qa   = assessor.assess(np.array(pil.convert("L")))
            p, _ = predict(model, device, pil)
            pos  = [DISEASE_LABELS[j] for j in range(14) if p[j] >= threshold]

            # Urgency flag
            urgency = "NORMAL"
            for label in pos:
                if URGENCY_MAP.get(label) == "URGENT":
                    urgency = "URGENT"; break
                elif URGENCY_MAP.get(label) == "PRIORITY":
                    urgency = "PRIORITY"

            results.append({
                "File":         f.name,
                "QA Score":     f"{qa.overall_score:.0f}/100",
                "QA Pass":      "✅" if qa.is_acceptable else "⚠️",
                "Detections":   ", ".join(pos) if pos else "No Finding",
                "Count":        len(pos),
                "Max Prob":     f"{float(np.max(p)):.4f}",
                "Urgency":      urgency
            })
            prog_bar.progress((i + 1) / len(batch_files))

        status_txt.markdown(
            "<div class='success-box'>✅ Batch processing complete!</div>",
            unsafe_allow_html=True
        )

        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

        # Summary stats
        n_urgent   = sum(1 for r in results if r["Urgency"] == "URGENT")
        n_priority = sum(1 for r in results if r["Urgency"] == "PRIORITY")
        n_normal   = sum(1 for r in results if r["Urgency"] == "NORMAL")
        n_qa_fail  = sum(1 for r in results if r["QA Pass"] == "⚠️")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Images",    len(results))
        c2.metric("⚡ Urgent",       n_urgent)
        c3.metric("⚠️ Priority",     n_priority)
        c4.metric("❌ QA Failed",    n_qa_fail)

        st.download_button(
            "⬇️ Download Batch Results (CSV)",
            data=df_results.to_csv(index=False),
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Model & Training Dashboard</div>",
                unsafe_allow_html=True)

    # Check for training log
    log_path = os.path.join(ROOT, "logs", f"training_{BACKBONE}.csv")

    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path)

        # Training curves
        fig_curves = go.Figure()
        fig_curves.add_trace(go.Scatter(
            x=df_log['epoch'], y=df_log['train_auroc'],
            mode='lines+markers', name='Train AUROC',
            line=dict(color='#00b4d8', width=2),
            marker=dict(size=5)
        ))
        fig_curves.add_trace(go.Scatter(
            x=df_log['epoch'], y=df_log['val_auroc'],
            mode='lines+markers', name='Val AUROC',
            line=dict(color='#3fb950', width=2),
            marker=dict(size=5)
        ))
        fig_curves.add_hline(
            y=0.5, line_dash='dash',
            line_color='rgba(248,81,73,0.4)',
            annotation_text='Random Baseline',
            annotation_font=dict(color='#f85149', size=9)
        )
        fig_curves.update_layout(
            title=dict(text='Training & Validation AUROC', font=dict(color='#e6edf3')),
            paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
            font=dict(color='#c9d1d9', family='JetBrains Mono'),
            xaxis=dict(title='Epoch', gridcolor='#21262d', color='#8b949e'),
            yaxis=dict(title='AUROC', gridcolor='#21262d', color='#8b949e',
                       range=[0, 1]),
            legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
            height=350
        )
        st.plotly_chart(fig_curves, use_container_width=True)

        # Loss curves
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=df_log['epoch'], y=df_log['train_loss'],
            mode='lines+markers', name='Train Loss',
            line=dict(color='#d29922', width=2), marker=dict(size=5)
        ))
        fig_loss.add_trace(go.Scatter(
            x=df_log['epoch'], y=df_log['val_loss'],
            mode='lines+markers', name='Val Loss',
            line=dict(color='#f85149', width=2), marker=dict(size=5)
        ))
        fig_loss.update_layout(
            title=dict(text='Training & Validation Loss', font=dict(color='#e6edf3')),
            paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
            font=dict(color='#c9d1d9', family='JetBrains Mono'),
            xaxis=dict(title='Epoch', gridcolor='#21262d', color='#8b949e'),
            yaxis=dict(title='Loss', gridcolor='#21262d', color='#8b949e'),
            legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
            height=300
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        # Stats row
        best_row = df_log.loc[df_log['val_auroc'].idxmax()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Val AUROC", f"{best_row['val_auroc']:.4f}",
                  f"Epoch {int(best_row['epoch'])}")
        c2.metric("Final Train Loss", f"{df_log['train_loss'].iloc[-1]:.4f}")
        c3.metric("Epochs Trained", len(df_log))
        c4.metric("Best Epoch", int(best_row['epoch']))

    else:
        st.markdown("""
        <div class='warn-box'>
        ⚠️ No training log found.<br>
        Run training first: <code>python src/training/train.py --epochs 5 --batch_size 8</code>
        </div>
        """, unsafe_allow_html=True)

    # Check for evaluation metrics
    metrics_path = os.path.join(ROOT, "outputs", "evaluation", "metrics.csv")
    if os.path.exists(metrics_path):
        st.markdown("---")
        st.markdown("<div class='section-header'>Test Set Evaluation Results</div>",
                    unsafe_allow_html=True)
        df_metrics = pd.read_csv(metrics_path)
        st.dataframe(df_metrics, use_container_width=True)

        # AUROC bar
        df_valid = df_metrics.dropna(subset=['AUROC']).sort_values('AUROC')
        colors   = ['#f85149' if v<0.7 else '#d29922' if v<0.8 else '#3fb950'
                    for v in df_valid['AUROC']]
        fig_bar  = go.Figure(go.Bar(
            x=df_valid['AUROC'], y=df_valid['Disease'],
            orientation='h', marker_color=colors,
            text=[f'{v:.3f}' for v in df_valid['AUROC']],
            textposition='outside',
            textfont=dict(size=9, color='#8b949e', family='JetBrains Mono')
        ))
        fig_bar.add_vline(x=0.5, line_dash='dash', line_color='rgba(248,81,73,0.5)')
        fig_bar.add_vline(x=df_valid['AUROC'].mean(),
                          line_dash='dot', line_color='#00b4d8',
                          annotation_text=f"Mean {df_valid['AUROC'].mean():.3f}",
                          annotation_font=dict(color='#00b4d8', size=9))
        fig_bar.update_layout(
            paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
            font=dict(color='#c9d1d9', family='JetBrains Mono'),
            xaxis=dict(range=[0.3, 1.05], gridcolor='#21262d'),
            yaxis=dict(tickfont=dict(size=9)),
            height=420, showlegend=False,
            title=dict(text='Per-Class AUROC (Test Set)',
                       font=dict(color='#e6edf3'))
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style='max-width:800px'>

    <div class='section-header'>Project Information</div>

    <div style='background:#161b22;border:1px solid #30363d;border-radius:8px;
                padding:1.2rem;margin-bottom:1rem'>
        <div style='font-size:0.9rem;font-weight:700;color:#e6edf3;margin-bottom:0.5rem'>
            Intelligent Multi-Stage Chest X-Ray Analysis System
        </div>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;
                    color:#8b949e;line-height:2'>
        Module: 7151CEM — Computing Individual Research Project<br>
        Student: Chenduluru Siva · MSc Computing<br>
        University: Coventry University<br>
        Supervisor: Dr. Diana Hintea
        </div>
    </div>

    <div class='section-header'>Pipeline Architecture</div>
    </div>
    """, unsafe_allow_html=True)

    pipeline_data = {
        "Stage": ["1", "2", "3", "4", "5"],
        "Component": [
            "Image Quality Assessment",
            "Multi-Label Classification",
            "Grad-CAM Explainability",
            "Attention Feature Maps",
            "Structured Report Generation"
        ],
        "Technology": [
            "OpenCV — Laplacian, brightness, contrast, black ratio",
            "DenseNet-121 fine-tuned on NIH ChestX-ray14 (14 classes)",
            "Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017)",
            "DenseBlock-4 channel activation visualisation",
            "Template-based NLG with urgency triage"
        ],
        "Reference": [
            "Original contribution",
            "Rajpurkar et al. (2017); Wang et al. (2017)",
            "Selvaraju et al. (2017)",
            "Original contribution",
            "Liu et al. (2019)"
        ]
    }
    st.dataframe(pd.DataFrame(pipeline_data), use_container_width=True)

    st.markdown("""
    <div class='section-header' style='margin-top:1.2rem'>Key References (APA 7)</div>
    <div style='background:#161b22;border:1px solid #30363d;border-radius:8px;
                padding:1.2rem;font-family:JetBrains Mono,monospace;
                font-size:0.68rem;color:#8b949e;line-height:2.2'>

    Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017).
    ChestX-ray8. <i style='color:#c9d1d9'>CVPR 2017</i>, 2097–2106.<br>

    Rajpurkar, P., et al. (2017). CheXNet.
    <i style='color:#c9d1d9'>arXiv:1711.05225</i><br>

    Irvin, J., et al. (2019). CheXpert.
    <i style='color:#c9d1d9'>AAAI 2019</i>, 33(1), 590–597.<br>

    Selvaraju, R. R., et al. (2017). Grad-CAM.
    <i style='color:#c9d1d9'>ICCV 2017</i>, 618–626.<br>

    Liu, F., et al. (2019). Clinically accurate chest X-ray report generation.
    <i style='color:#c9d1d9'>MLHC 2019</i>, 249, 249–269.

    </div>

    <div class='warn-box' style='margin-top:1rem'>
    ⚠️ <b>DISCLAIMER:</b> This is a research prototype for academic purposes only
    (7151CEM MSc dissertation). It is NOT validated for clinical use.
    All outputs must be reviewed and verified by a qualified radiologist.
    </div>
    """, unsafe_allow_html=True)