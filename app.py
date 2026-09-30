"""Streamlit web app: upload or photograph a leaf, get a diagnosis.

Run locally:   streamlit run app.py
"""

import json

import streamlit as st
from PIL import Image

from plant_disease import config
from plant_disease.inference import PlantDiseasePredictor, parse_label

LOW_CONFIDENCE = 0.5

st.set_page_config(page_title="Plant disease detector", page_icon="🌿", layout="centered")


@st.cache_resource(show_spinner="Loading model ...")
def load_predictor() -> PlantDiseasePredictor:
    return PlantDiseasePredictor.from_dir(config.MODEL_DIR)


def load_model_info() -> dict:
    path = config.MODEL_DIR / config.HISTORY_FILENAME
    try:
        return json.loads(path.read_text()).get("config", {})
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------ header
st.title("🌿 Plant disease detector")
st.write("Upload a photo of a single leaf. The model recognises 38 healthy and diseased "
         "classes across 14 crops, and tells you whether the plant looks healthy.")

try:
    predictor = load_predictor()
except FileNotFoundError as err:
    st.error(f"No trained model found.\n\n{err}\n\n"
             "Train one with `python train.py`, or copy `plant_disease_model.keras` and "
             "`class_names.json` into the `models/` folder.")
    st.stop()

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About the model")
    info = load_model_info()
    if info:
        st.write(f"Architecture: **{info.get('arch', 'cnn')}**")
        if "best_val_accuracy" in info:
            st.write(f"Validation accuracy: **{info['best_val_accuracy']:.1%}**")
    st.write(f"Input size: {predictor.img_size[0]}×{predictor.img_size[1]} px")

    crops = sorted({parse_label(name)[0] for name in predictor.class_names})
    st.write("Supported crops: " + ", ".join(crops) + ".")

    st.caption("Trained on the PlantVillage-based *New Plant Diseases Dataset*: close-up leaves "
               "on plain backgrounds. Field photos with busy backgrounds may be less accurate. "
               "This is a learning project, not a substitute for an agronomist.")

# ------------------------------------------------------------------- input
upload_tab, camera_tab = st.tabs(["Upload a photo", "Use camera"])
with upload_tab:
    uploaded = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png", "webp"],
                                label_visibility="collapsed")
with camera_tab:
    snapshot = st.camera_input("Take a photo of a leaf", label_visibility="collapsed")

source = uploaded or snapshot
if source is None:
    st.info("Choose a leaf image to get a diagnosis.")
    st.stop()

try:
    image = Image.open(source).convert("RGB")
except Exception:  # noqa: BLE001 - any decode failure
    st.error("That file couldn't be read as an image. Try a JPG or PNG.")
    st.stop()

# ------------------------------------------------------------------ result
with st.spinner("Analysing leaf ..."):
    pred = predictor.predict(image, top_k=3)

left, right = st.columns([1, 1], gap="large")
with left:
    st.image(image)

with right:
    if pred.is_healthy:
        st.success(f"**Healthy**  \n{pred.plant} leaf with no disease detected.")
    else:
        st.error(f"**{pred.condition}**  \nDisease detected on a {pred.plant} leaf.")
    st.metric("Confidence", f"{pred.confidence:.1%}")

    if pred.confidence < LOW_CONFIDENCE:
        st.warning("Low confidence. Try a sharper, closer photo of one leaf on a plain background.")

    st.subheader("Top predictions")
    for name, prob in pred.top_k:
        plant, condition, _ = parse_label(name)
        st.write(f"{plant}: {condition}")
        st.progress(min(max(prob, 0.0), 1.0), text=f"{prob:.1%}")
