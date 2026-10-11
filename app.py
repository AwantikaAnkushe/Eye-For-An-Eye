"""Local image-screening interface for the retinal severity model."""

# /// script
# requires-python = ">=3.10"
# dependencies = ["streamlit", "torch", "torchvision"]
# ///

import hashlib
import io
import csv
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models

from BlindnessDetection import CLASS_COUNT, DEFAULT_DATASET_DIR, IMAGE_SIZE, build_transforms


GRADE_LABELS = (
    "No diabetic retinopathy",
    "Mild non-proliferative retinopathy",
    "Moderate non-proliferative retinopathy",
    "Severe non-proliferative retinopathy",
    "Proliferative diabetic retinopathy",
)
CHECKPOINT_PATH = Path(__file__).parent / "outputs" / "best_model.pt"


@st.cache_data
def load_dataset_labels(csv_path: str) -> dict[str, int]:
    labels_path = Path(csv_path)
    if not labels_path.is_file():
        return {}
    with labels_path.open(newline="", encoding="utf-8-sig") as label_file:
        reader = csv.DictReader(label_file)
        if not {"image", "level"}.issubset(reader.fieldnames or []):
            return {}
        return {row["image"].strip(): int(row["level"]) for row in reader}


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model(checkpoint_path: str) -> tuple[nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("class_count") != CLASS_COUNT:
        raise ValueError("The checkpoint does not contain the expected five classes.")
    if checkpoint.get("image_size") != IMAGE_SIZE:
        raise ValueError("The checkpoint image size does not match this application.")

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, CLASS_COUNT)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def predict(image: Image.Image, model: nn.Module, device: torch.device) -> list[float]:
    image_tensor = build_transforms(training=False)(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores = torch.softmax(model(image_tensor)[0], dim=0)
    return scores.cpu().tolist()


st.set_page_config(
    page_title="Eye for an Eye",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172d2b;
        --muted: #63716b;
        --paper: #f4f5ef;
        --line: #d8dfd6;
        --green: #22584c;
        --lime: #cce16c;
        --coral: #bf5d45;
    }
    [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    .main .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero {
        padding: 2rem 2.2rem 2.1rem;
        color: #f4f5ef;
        background-color: #173c35;
        background-image: repeating-linear-gradient(135deg, transparent 0 22px, #ffffff08 22px 23px);
        border-radius: 5px;
        margin-bottom: 2rem;
    }
    .hero-kicker, .section-kicker {
        color: var(--lime);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: #f4f5ef;
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 3.5rem;
        font-weight: 500;
        line-height: 1.08;
        margin: 0.55rem 0;
    }
    .hero p { color: #d2ddd4; max-width: 660px; margin: 0; font-size: 1rem; }
    h2, h3 { color: var(--ink); }
    [data-testid="stFileUploader"] section {
        background: #fbfcf8;
        border: 1px dashed #91a49a;
        border-radius: 5px;
    }
    [data-testid="stProgress"] > div > div { background: var(--green); }
    [data-testid="stProgress"] > div > div > div {
        color: #ffffff !important;
        font-weight: 600;
    }
    [data-testid="stProgress"] label,
    [data-testid="stProgress"] .stProgress-label,
    [data-testid="stProgress"] div[role="progressbar"] + div {
        color: #ffffff !important;
        font-weight: 600;
    }
    div.stButton > button[kind="primary"] {
        background: var(--green);
        border-color: var(--green);
        border-radius: 4px;
        min-height: 2.8rem;
    }
    .result-grade {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 1.7rem;
        line-height: 1.2;
        color: var(--ink);
        padding: 0.7rem 0 1rem;
        border-bottom: 1px solid var(--line);
    }
    .disclaimer {
        border-left: 3px solid var(--coral);
        color: #465650;
        padding: 0.25rem 0 0.25rem 0.8rem;
        font-size: 0.9rem;
    }
    .footer { color: var(--muted); border-top: 1px solid var(--line); padding-top: 1rem; margin-top: 2.5rem; font-size: 0.82rem; }
    @media (max-width: 700px) {
        .main .block-container { padding: 1rem 1rem 2rem; }
        .hero { padding: 1.4rem; }
        .hero h1 { font-size: 2.25rem; }
    }
    @media (max-width: 380px) {
        .hero h1 { font-size: 2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <header class="hero">
      <div class="hero-kicker">Eye for an eye</div>
      <h1>A closer look at retinal health.</h1>
      <p>Diabetic retinopathy severity estimate from a retinal fundus photograph.</p>
    </header>
    """,
    unsafe_allow_html=True,
)

if CHECKPOINT_PATH.is_file():
    st.caption(f"Model checkpoint: `{CHECKPOINT_PATH.relative_to(Path(__file__).parent)}`")
else:
    st.warning(
        "The model has not been trained yet. Known dataset images can show their stored labels; "
        "other images need a trained model.",
        icon="⚠️",
    )
    st.code("& .\\.venv\\Scripts\\python.exe .\\BlindnessDetection.py", language="powershell")

upload_column, result_column = st.columns([1, 1], gap="large")
with upload_column:
    st.markdown('<div class="section-kicker">01 / Retinal image</div>', unsafe_allow_html=True)
    st.subheader("Choose a fundus photograph")
    uploaded_image = st.file_uploader(
        "Upload one eye image",
        type=("jpg", "jpeg", "png", "webp"),
        help="Use a clear color retinal fundus image. Images are processed locally and are not saved.",
    )

with result_column:
    st.markdown('<div class="section-kicker">02 / Model output</div>', unsafe_allow_html=True)
    st.subheader("Severity estimate")

if uploaded_image is None:
    with result_column:
        st.info("Upload a retinal image to view the model output.")
else:
    image_bytes = uploaded_image.getvalue()
    image_key = hashlib.sha256(image_bytes).hexdigest()
    dataset_grade = None
    if not CHECKPOINT_PATH.is_file():
        dataset_labels = load_dataset_labels(
            str(DEFAULT_DATASET_DIR / "trainLabels.csv")
        )
        dataset_grade = dataset_labels.get(Path(uploaded_image.name).stem)
    try:
        retinal_image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    except (OSError, Image.DecompressionBombError):
        retinal_image = None
        with upload_column:
            st.error("This file could not be opened as an image. Try a JPG or PNG fundus photo.")

    if retinal_image is not None:
        with upload_column:
            st.image(retinal_image, caption=uploaded_image.name, use_container_width=True)
            checkpoint_exists = CHECKPOINT_PATH.is_file()
            known_dataset_sample = dataset_grade is not None
            analyze_disabled = not checkpoint_exists and not known_dataset_sample
            analyze_clicked = st.button(
                "Analyze image",
                type="primary",
                use_container_width=True,
                disabled=analyze_disabled,
            )
            if analyze_disabled:
                st.caption(
                    r"Disabled until the model is trained. Run `& .\.venv\Scripts\python.exe "
                    r".\BlindnessDetection.py` in PowerShell, then refresh this page."
                )
            elif not checkpoint_exists:
                st.caption(
                    "This filename matches a labeled dataset image. Clicking shows its stored "
                    "label, not a model prediction."
                )

        if analyze_clicked:
            try:
                if checkpoint_exists:
                    model, device = load_model(str(CHECKPOINT_PATH))
                    with st.spinner("Analyzing retinal image..."):
                        model_scores = predict(retinal_image, model, device)
                    st.session_state["screening_result"] = {
                        "image_key": image_key,
                        "source": "model",
                        "scores": model_scores,
                    }
                else:
                    st.session_state["screening_result"] = {
                        "image_key": image_key,
                        "source": "dataset",
                        "grade": dataset_grade,
                    }
            except (KeyError, RuntimeError, ValueError, OSError) as error:
                st.session_state.pop("screening_result", None)
                with result_column:
                    st.error(f"The model could not analyze this image: {error}")

        result = st.session_state.get("screening_result")
        if result and result["image_key"] == image_key:
            with result_column:
                if result["source"] == "dataset":
                    st.caption("Stored dataset label · not a model prediction")
                    st.markdown(
                        f'<div class="result-grade">Grade {result["grade"]}: '
                        f'{GRADE_LABELS[result["grade"]]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    scores = result["scores"]
                    predicted_grade = max(range(CLASS_COUNT), key=scores.__getitem__)
                    st.markdown(
                        f'<div class="result-grade">{GRADE_LABELS[predicted_grade]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption("Model output by training-set grade")
                    for grade, score in enumerate(scores):
                        st.markdown(
                            f"""
                            <div style="margin: 0.7rem 0 0.9rem;">
                              <div style="color: #000000; font-weight: 600; margin-bottom: 0.35rem; font-size: 0.95rem;">
                                Grade {grade} · {GRADE_LABELS[grade]} · {score * 100:.1f}%
                              </div>
                              <div style="width: 100%; height: 20px; background: #dfe6df; border-radius: 6px; overflow: hidden;">
                                <div style="width: {score * 100:.1f}%; height: 100%; background: #22584c; border-radius: 6px;"></div>
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        '<p class="disclaimer">These scores are not clinically calibrated probabilities. '
                        "This tool cannot diagnose disease or determine whether someone will become blind. "
                        "Discuss results with an eye-care professional; seek urgent care for sudden vision changes.</p>",
                        unsafe_allow_html=True,
                    )
        elif not CHECKPOINT_PATH.is_file() and dataset_grade is None:
            with result_column:
                st.caption("The upload is ready. Train the model to enable analysis.")

st.markdown(
    '<div class="footer">Research demonstration only · Not a medical device or a substitute for an eye examination.</div>',
    unsafe_allow_html=True,
)