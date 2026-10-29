"""Local Streamlit dashboard for the EdgeSat AI TorchScript demo."""
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "artifacts" / "model_config.json"
VARIANTS = {
    "Noise-trained": ROOT / "artifacts" / "noise" / "qacnet_torchscript.pt",
    "Clean-only": ROOT / "artifacts" / "clean" / "qacnet_torchscript.pt",
}

st.set_page_config(page_title="EdgeSat AI | Tile Viewer", page_icon="🛰️", layout="wide")
st.title("🛰️ EdgeSat AI")
st.caption("Satellite tile land-cover demo · QAC-Net · local inference")
st.info(
    "This prototype predicts EuroSAT land cover and clean vs synthetic corruption. "
    "It does not detect clouds. Entropy below is an uncalibrated ambiguity indicator."
)

@st.cache_resource
def load_model(model_path: str):
    return torch.jit.load(model_path, map_location="cpu").eval()

model_choice = st.selectbox(
    "Model variant",
    list(VARIANTS),
    help="Noise-trained is intended to be more robust to synthetic corruption; clean-only is the baseline.",
)
model_path = VARIANTS[model_choice]
if not model_path.exists():
    st.error(f"Model file not found: {model_path}")
    st.stop()
if not CONFIG_PATH.exists():
    st.error(f"Class configuration not found: {CONFIG_PATH}")
    st.stop()

import json
config = json.loads(CONFIG_PATH.read_text())

uploaded = st.file_uploader(
    "Upload a satellite image tile", type=["jpg", "jpeg", "png", "tif", "tiff"],
)
if uploaded is None:
    st.markdown("Upload a tile to see its predictions. The model resizes it to **64 × 64 RGB**.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()

model = load_model(str(model_path))
tensor = Compose([Resize((64, 64)), ToTensor()])(image).unsqueeze(0)
with torch.inference_mode():
    landcover_logits, quality_logits = model(tensor)
    landcover_prob = landcover_logits.softmax(1)[0]
    quality_prob = quality_logits.softmax(1)[0]

class_id = int(landcover_prob.argmax())
quality_id = int(quality_prob.argmax())
tiny = torch.finfo(landcover_prob.dtype).tiny
landcover_entropy = -(landcover_prob * landcover_prob.clamp_min(tiny).log()).sum()
quality_entropy = -(quality_prob * quality_prob.clamp_min(tiny).log()).sum()
landcover_uncertainty = float(landcover_entropy / torch.log(torch.tensor(float(landcover_prob.numel()))))
quality_uncertainty = float(quality_entropy / torch.log(torch.tensor(float(quality_prob.numel()))))
class_names = config["classes"]
quality_names = ["Clean", "Synthetically corrupted"]

left, right = st.columns([1, 1.35], gap="large")
with left:
    st.image(image, caption=f"Uploaded tile · original size {image.width} × {image.height}", use_container_width=True)
with right:
    st.subheader("Prediction")
    col1, col2 = st.columns(2)
    col1.metric("Land cover", class_names[class_id])
    col2.metric("Top-class score", f"{float(landcover_prob[class_id]) * 100:.1f}%")
    qcol1, qcol2 = st.columns(2)
    qcol1.metric("Input quality", quality_names[quality_id])
    qcol2.metric("Quality score", f"{float(quality_prob[quality_id]) * 100:.1f}%")

    st.markdown("**Land-cover scores**")
    for idx in torch.argsort(landcover_prob, descending=True).tolist():
        score = float(landcover_prob[idx])
        st.progress(score, text=f"{class_names[idx]} · {score * 100:.1f}%")

st.divider()
st.subheader("Prediction ambiguity")
st.caption("Normalized softmax entropy: 0 means concentrated scores; values nearer 1 mean more spread-out scores.")
uc1, uc2 = st.columns(2)
with uc1:
    st.write(f"**Land-cover entropy:** {landcover_uncertainty:.3f} / 1.000")
    st.progress(landcover_uncertainty)
with uc2:
    st.write(f"**Quality entropy:** {quality_uncertainty:.3f} / 1.000")
    st.progress(quality_uncertainty)
st.warning(
    "These entropy values are not calibrated probabilities that the prediction is wrong. "
    "The quality head was trained on synthetic corruption labels, not cloud labels."
)
