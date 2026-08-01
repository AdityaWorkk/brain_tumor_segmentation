import os
import random
import cv2
import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import streamlit as st
from pathlib import Path
import gdown

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Brain MRI Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. SETUP & CONSTANTS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GDRIVE_FILE_ID = st.secrets.get("GDRIVE_FILE_ID")
MODEL_FILENAME = st.secrets.get("MODEL_FILENAME", "best_brain_tumor_unet_2.pth")
SAMPLES_DIR = Path("static/test_samples")
MAX_GENERATIONS = 15
MAX_RANDOM = 15

# ==========================================
# 3. MODEL LOADING (cached)
# ==========================================
@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_FILENAME):
        model_path = MODEL_FILENAME
    else:
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            model_path = gdown.download(url, MODEL_FILENAME, quiet=False)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)

    model.eval()
    return model

model = load_trained_model()

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def get_sample_files():
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
    files = []
    for ext in extensions:
        files.extend(list(SAMPLES_DIR.glob(ext)))
    return files

def load_random_sample():
    files = get_sample_files()
    if not files:
        return None
    chosen = random.choice(files)
    img = Image.open(chosen).convert("RGB")
    img.thumbnail((512, 512), Image.LANCZOS)
    return np.array(img)

def run_segmentation(image):
    img_np = np.array(image) if not isinstance(image, np.ndarray) else image
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]

    augmented = transform(image=img_np)
    tensor = augmented['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        pred_binary = (probs > 0.5).float().squeeze().cpu().numpy()

    mask_uint8 = (pred_binary * 255).astype(np.uint8)
    has_tumor = bool(pred_binary.any())
    return mask_uint8, has_tumor

# ==========================================
# 5. SESSION STATE INIT
# ==========================================
if 'remaining_gen' not in st.session_state:
    st.session_state.remaining_gen = MAX_GENERATIONS
if 'random_count' not in st.session_state:
    st.session_state.random_count = 0
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'mask_image' not in st.session_state:
    st.session_state.mask_image = None
if 'tumor_label' not in st.session_state:
    st.session_state.tumor_label = None
if 'source_label' not in st.session_state:
    st.session_state.source_label = None

# ==========================================
# 6. UI
# ==========================================
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    h1, h2, h3 { color: #60a5fa !important; }
    .disclaimer {
        background: rgba(59, 130, 246, 0.1); border: 1px solid #3b82f6;
        padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;
        color: #93c5fd; font-size: 0.9rem;
    }
    .positive {
        background: rgba(239, 68, 68, 0.15); color: #fca5a5;
        border: 2px solid #ef4444; padding: 10px 20px;
        border-radius: 8px; font-weight: 700; text-align: center; font-size: 1.1rem;
    }
    .negative {
        background: rgba(34, 197, 94, 0.15); color: #86efac;
        border: 2px solid #22c55e; padding: 10px 20px;
        border-radius: 8px; font-weight: 700; text-align: center; font-size: 1.1rem;
    }
    div[data-testid="stToolbar"] { display: none; }
    footer { display: none; }
    button[kind="primary"] { background-color: #3b82f6 !important; }
    @media (prefers-color-scheme: light) {
        h1, h2, h3 { color: #1d4ed8 !important; }
        .disclaimer { color: #1e40af; }
    }
</style>
""", unsafe_allow_html=True)

# Header
col_header, col_device = st.columns([4, 1])
with col_header:
    st.markdown("# 🧠 Brain MRI Tumor Segmentation Studio")
    st.markdown(f"U-Net (ResNet34)  \|  70.1% Dice  \|  Device: **{str(DEVICE).upper()}**  \|  Generations left: **{st.session_state.remaining_gen}/{MAX_GENERATIONS}**")

# Disclaimer
st.markdown(
    '<div class="disclaimer">'
    '⚠️ <strong>Medical Disclaimer:</strong> This model has ~70% Dice accuracy. '
    'For research/demo only. <strong>Not for medical diagnosis.</strong>'
    '</div>',
    unsafe_allow_html=True
)

# Main columns
col_input, col_output = st.columns([1, 2], gap="large")

# ---- LEFT COLUMN: INPUT ----
with col_input:
    st.subheader("1. Provide MRI Image")

    uploaded_file = st.file_uploader(
        "Upload MRI Scan",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img.thumbnail((512, 512), Image.LANCZOS)
        st.session_state.current_image = np.array(img)
        st.session_state.source_label = f"Uploaded: {uploaded_file.name}"
        st.session_state.mask_image = None
        st.session_state.tumor_label = None

    st.markdown("---")

    col_rand1, col_rand2 = st.columns([3, 1])
    with col_rand1:
        rand_disabled = st.session_state.random_count >= MAX_RANDOM
        if st.button(
            f"🎲 Random Image ({st.session_state.random_count}/{MAX_RANDOM})",
            disabled=rand_disabled,
            use_container_width=True
        ):
            img = load_random_sample()
            if img is not None:
                st.session_state.current_image = img
                st.session_state.source_label = "Random MRI sample"
                st.session_state.mask_image = None
                st.session_state.tumor_label = None
                st.session_state.random_count += 1

    st.markdown("---")

    pred_disabled = (
        st.session_state.current_image is None
        or st.session_state.remaining_gen <= 0
    )
    if st.button(
        "🔬 Run Tumor Segmentation",
        disabled=pred_disabled,
        use_container_width=True,
        type="primary"
    ):
        with st.spinner("Analyzing brain MRI..."):
            mask, has_tumor = run_segmentation(st.session_state.current_image)
            st.session_state.mask_image = mask
            st.session_state.tumor_label = "POSITIVE" if has_tumor else "NEGATIVE"
            st.session_state.remaining_gen -= 1

# ---- RIGHT COLUMN: OUTPUT ----
with col_output:
    st.subheader("2. Output Analysis")

    col_img1, col_img2 = st.columns(2)

    with col_img1:
        if st.session_state.current_image is not None:
            st.image(
                st.session_state.current_image,
                caption=st.session_state.source_label or "Source Scan",
                use_container_width=True
            )
        else:
            st.info("Upload an image or generate a random sample")

    with col_img2:
        if st.session_state.mask_image is not None:
            st.image(
                st.session_state.mask_image,
                caption="Predicted Mask",
                use_container_width=True
            )

            if st.session_state.tumor_label == "POSITIVE":
                st.markdown(
                    '<div class="positive">Tumor: POSITIVE</div>',
                    unsafe_allow_html=True
                )
            elif st.session_state.tumor_label == "NEGATIVE":
                st.markdown(
                    '<div class="negative">Tumor: NEGATIVE</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Predicted mask will appear here")

    # Model info
    with st.expander("Model Information"):
        st.markdown(f"""
        - **Architecture:** U-Net (ResNet34)
        - **Input Resolution:** 256×256
        - **Validation Dice Score:** 70.1%
        - **System Device:** {str(DEVICE).upper()}
        - **Rate Limit:** {MAX_GENERATIONS} predictions per session
        """)
