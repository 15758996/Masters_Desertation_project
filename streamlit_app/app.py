"""
Streamlit UI — Intelligent Multi-Stage Chest X-Ray Analysis System
Author: Chenduluru Siva | 7151CEM
Run: streamlit run streamlit_app/app.py
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
import cv2

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
    page_title="ChestAI — X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-title{font-size:2rem;font-weight:700;color:#1a3a5c}
.subtitle  {font-size:1rem;color:#555;margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model …")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(BACKBONE).to(device)
    ckpt   = os.path.join(SAVED_MODELS_DIR, f"best_{BACKBONE}.pt")
    if os.path.exists(ckpt):
        model  = load_checkpoint(model, ckpt, device)
        status = f"✅ Loaded best_{BACKBONE}.pt"
    else:
        status = "⚠️ No checkpoint found — demo mode (random weights)"
    return model, device, status


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
    sorted_idx = np.argsort(probs)[::-1][:top_k]
    gcam       = GradCAM(model)
    orig_np    = denormalise(tensor.squeeze(0))
    fig, axes  = plt.subplots(1, top_k + 1, figsize=(5*(top_k+1), 5))
    axes[0].imshow(orig_np); axes[0].set_title("Original"); axes[0].axis("off")
    for col, idx in enumerate(sorted_idx, start=1):
        hm = gcam.generate(tensor.to(device), int(idx))
        ov = gcam.overlay(orig_np, hm)
        axes[col].imshow(ov)
        axes[col].set_title(f"{DISEASE_LABELS[idx]}\np={probs[idx]:.3f}", fontsize=9)
        axes[col].axis("off")
    plt.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    threshold    = st.slider("Detection threshold", 0.1, 0.9,
                             float(CLASSIFICATION_THRESHOLD), 0.05)
    show_gradcam = st.checkbox("Generate Grad-CAM", value=True)
    top_k        = st.slider("Grad-CAM top-K diseases", 1, 6, 4)
    st.markdown("---")
    model, device, model_status = load_model()
    st.markdown(f"**Model:** {model_status}")
    st.markdown(f"**Device:** {'GPU 🟢' if device.type=='cuda' else 'CPU 🟡'}")
    st.markdown("---")
    st.markdown("**Pipeline Stages**")
    st.markdown("1️⃣ Quality Assessment  \n2️⃣ Disease Classification  \n"
                "3️⃣ Grad-CAM Explainability  \n4️⃣ Structured Report")


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🫁 Intelligent Chest X-Ray Analysis System</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-Stage: Quality Assessment → Classification → '
            'Explainability → Structured Report</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔬 Single Image", "📦 Batch Analysis", "📖 About"])

# ══ TAB 1: SINGLE IMAGE ══════════════════════════════════════════════════════
with tab1:
    uploaded = st.file_uploader(
        "Upload a Chest X-Ray (PNG / JPG)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")
        assessor  = get_assessor()
        reporter  = get_reporter()

        # Stage 1 — QA
        st.markdown("---")
        st.markdown("### 📋 Stage 1 — Image Quality Assessment")
        img_gray = np.array(pil_image.convert("L"))
        qa       = assessor.assess(img_gray)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Quality Score",  f"{qa.overall_score}/100")
        c2.metric("Brightness",     f"{qa.brightness:.0f}")
        c3.metric("Contrast",       f"{qa.contrast:.0f}")
        c4.metric("Sharpness",      f"{qa.sharpness:.0f}")
        c5.metric("Black Ratio",    f"{qa.black_ratio:.2%}")

        if qa.is_acceptable:
            st.success(f"✅ {qa.recommendation}")
        else:
            st.error(f"⚠️ {qa.recommendation}")
            for issue in qa.issues:
                st.warning(f"Issue: {issue}")

        # Stage 2 — Classification
        st.markdown("---")
        st.markdown("### 🧠 Stage 2 — Disease Classification")
        with st.spinner("Running DenseNet-121 …"):
            probs, tensor = predict(model, device, pil_image)

        col_img, col_pred = st.columns([1, 2])
        with col_img:
            st.image(pil_image, caption=uploaded.name, use_container_width=True)

        with col_pred:
            positive = [(DISEASE_LABELS[i], probs[i])
                        for i in range(14) if probs[i] >= threshold]
            positive.sort(key=lambda x: -x[1])

            if positive:
                st.markdown("**Detected conditions:**")
                for label, p in positive:
                    st.markdown(f"**{label}** — `{p:.3f}`")
                    st.progress(float(p))
            else:
                st.success("**No significant findings** above threshold.")

            with st.expander("📊 All 14 disease probabilities"):
                import pandas as pd
                df_show = pd.DataFrame({
                    "Disease":     DISEASE_LABELS,
                    "Probability": [f"{p:.4f}" for p in probs],
                    "Detected":    ["✅" if p >= threshold else "—" for p in probs]
                })
                st.dataframe(df_show, use_container_width=True)

        # Stage 3 — Grad-CAM
        if show_gradcam:
            st.markdown("---")
            st.markdown("### 🗺️ Stage 3 — Grad-CAM Explainability")
            st.caption("Heatmaps highlight regions most influential for each prediction.")
            with st.spinner("Generating Grad-CAM …"):
                try:
                    fig = make_gradcam_figure(model, tensor, probs, device, top_k)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"Grad-CAM unavailable: {e}")

        # Stage 4 — Report
        st.markdown("---")
        st.markdown("### 📄 Stage 4 — Structured Report")
        report = reporter.generate(
            probs=probs, filename=uploaded.name,
            qa_score=qa.overall_score, qa_acceptable=qa.is_acceptable
        )
        with st.expander("📋 View Full Report", expanded=True):
            st.code(report.full_text, language=None)
        st.download_button(
            "⬇️ Download Report (.txt)",
            data=report.full_text,
            file_name=f"{report.report_id}.txt",
            mime="text/plain"
        )

    else:
        st.info("👆 Upload a chest X-ray image to begin analysis.")
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. Upload a frontal chest X-ray PNG or JPG
            2. The system runs all 4 stages automatically
            3. Adjust threshold in the sidebar to control sensitivity
            4. Download the structured report as a text file
            > ⚠️ Research prototype — not for clinical use.
            """)


# ══ TAB 2: BATCH ═════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📦 Batch Analysis")
    batch_files = st.file_uploader(
        "Upload multiple X-rays",
        type=["png","jpg","jpeg"],
        accept_multiple_files=True
    )
    if batch_files:
        assessor = get_assessor()
        results  = []
        bar      = st.progress(0)
        status   = st.empty()
        for i, f in enumerate(batch_files):
            status.text(f"Analysing {f.name} ({i+1}/{len(batch_files)}) …")
            pil  = Image.open(f).convert("RGB")
            qa   = assessor.assess(np.array(pil.convert("L")))
            p, _ = predict(model, device, pil)
            pos  = [DISEASE_LABELS[j] for j in range(14) if p[j] >= threshold]
            results.append({
                "File":       f.name,
                "QA Score":   qa.overall_score,
                "QA Pass":    "✅" if qa.is_acceptable else "❌",
                "Detections": ", ".join(pos) if pos else "No Finding",
                "Max Prob":   f"{float(np.max(p)):.3f}"
            })
            bar.progress((i+1)/len(batch_files))
        status.text("✅ Batch complete!")
        import pandas as pd
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)
        st.download_button(
            "⬇️ Download CSV",
            data=df_results.to_csv(index=False),
            file_name="batch_results.csv",
            mime="text/csv"
        )


# ══ TAB 3: ABOUT ═════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    ## 📖 Project Information
    **Module:** 7151CEM — Computing Individual Research Project
    **Student:** Chenduluru Siva | MSc Computing | Coventry University
    **Supervisor:** Dr. Diana Hintea

    ### Pipeline
    | Stage | Component | Method |
    |-------|-----------|--------|
    | 1 | Quality Assessment | OpenCV heuristics |
    | 2 | Classification | DenseNet-121 (NIH ChestX-ray14) |
    | 3 | Explainability | Grad-CAM |
    | 4 | Reporting | Template-based NLG |

    ### Dataset
    **NIH ChestX-ray14** — 112,120 frontal X-rays, 14 disease labels, 30,805 patients.

    ### References
    - Rajpurkar et al. (2017). CheXNet.
    - Wang et al. (2017). ChestX-ray8.
    - Selvaraju et al. (2017). Grad-CAM.
    - Irvin et al. (2019). CheXpert.

    > ⚠️ Research prototype only. Not validated for clinical use.
    """)