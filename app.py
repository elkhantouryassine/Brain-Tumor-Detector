import time
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing import image as keras_image


# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="MediScan AI | Diagnostic IRM",
    layout="wide",
    initial_sidebar_state="expanded"
)

IMG_SIZE = 224
DISPLAY_WIDTH = 340
MODEL_PATH = "densenet_brain_tumor.h5"


# ============================================================
# THEME MEDICAL MODERNE - AERO MEDICAL
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --primary: #0ea5e9;
        --primary-dark: #0284c7;
        --primary-light: #e0f2fe;
        --secondary: #6366f1;
        --accent: #06b6d4;
        --success: #10b981;
        --success-light: #d1fae5;
        --danger: #ef4444;
        --danger-light: #fee2e2;
        --warning: #f59e0b;
        --warning-light: #fef3c7;
        --dark: #0f172a;
        --gray-900: #1e293b;
        --gray-700: #334155;
        --gray-500: #64748b;
        --gray-300: #cbd5e1;
        --gray-100: #f1f5f9;
        --white: #ffffff;
        --glass: rgba(255, 255, 255, 0.7);
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0f9ff 100%);
        background-attachment: fixed;
    }

    #MainMenu, header, footer {
        display: none;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Animations */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse-ring {
        0% { transform: scale(0.8); opacity: 0.5; }
        100% { transform: scale(1.3); opacity: 0; }
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
    }

    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }

    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        14% { transform: scale(1.05); }
        28% { transform: scale(1); }
        42% { transform: scale(1.05); }
        70% { transform: scale(1); }
    }

    /* Sidebar moderne */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: none;
    }

    section[data-testid="stSidebar"] * {
        color: var(--white);
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--white);
        font-weight: 700;
    }

    section[data-testid="stSidebar"] [data-testid="stSlider"] label {
        color: #94a3b8 !important;
    }

    section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
        background: var(--primary) !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] .stCheckbox label {
        color: #cbd5e1 !important;
    }

    /* Header premium */
    .aero-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.6));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.5);
        border-radius: 24px;
        padding: 2rem 2.5rem;
        box-shadow: 
            0 4px 6px -1px rgba(0, 0, 0, 0.05),
            0 20px 40px -10px rgba(14, 165, 233, 0.15);
        position: relative;
        overflow: hidden;
        animation: slideUp 0.6s ease-out;
    }

    .aero-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary), var(--secondary), var(--accent));
    }

    .aero-header::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(14, 165, 233, 0.1), transparent 70%);
        border-radius: 50%;
    }

    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, var(--primary-light), #dbeafe);
        color: var(--primary-dark);
        border: 1px solid rgba(14, 165, 233, 0.2);
        border-radius: 100px;
        padding: 0.4rem 1rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .header-badge .dot {
        width: 8px;
        height: 8px;
        background: var(--success);
        border-radius: 50%;
        animation: heartbeat 2s infinite;
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--dark);
        letter-spacing: -0.04em;
        line-height: 1.1;
        margin-bottom: 0.5rem;
    }

    .header-subtitle {
        color: var(--gray-500);
        font-size: 1.05rem;
        max-width: 700px;
        line-height: 1.7;
    }

    /* Alertes modernes */
    .modern-alert {
        background: linear-gradient(135deg, #fffbeb, #fef3c7);
        border: 1px solid #fcd34d;
        border-radius: 16px;
        padding: 1rem 1.25rem;
        display: flex;
        align-items: center;
        gap: 12px;
        animation: slideUp 0.5s ease-out 0.2s both;
        margin: 1.5rem 0;
    }

    .modern-alert-icon {
        width: 36px;
        height: 36px;
        background: #f59e0b;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.2rem;
        flex-shrink: 0;
    }

    .modern-alert-text {
        color: #92400e;
        font-size: 0.9rem;
        line-height: 1.6;
        font-weight: 500;
    }

    /* Cartes glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.6);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 
            0 4px 6px -1px rgba(0, 0, 0, 0.05),
            0 10px 20px -5px rgba(0, 0, 0, 0.05);
        animation: slideUp 0.5s ease-out;
    }

    .glass-card:hover {
        box-shadow: 
            0 10px 15px -3px rgba(0, 0, 0, 0.08),
            0 20px 30px -10px rgba(14, 165, 233, 0.1);
        transition: all 0.3s ease;
    }

    /* Upload zone */
    .upload-zone {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border: 2px dashed #cbd5e1;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .upload-zone:hover {
        border-color: var(--primary);
        background: linear-gradient(135deg, #e0f2fe, #f0f9ff);
    }

    /* Boutons premium */
    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 14px;
        border: none;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
        box-shadow: 0 10px 25px -5px rgba(14, 165, 233, 0.4);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px -5px rgba(14, 165, 233, 0.5);
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Bouton secondaire */
    .stButton > button[kind="secondary"] {
        background: var(--white);
        color: var(--gray-700);
        border: 2px solid var(--gray-300);
        box-shadow: none;
    }

    .stButton > button[kind="secondary"]:hover {
        border-color: var(--primary);
        color: var(--primary);
    }

    /* Images */
    div[data-testid="stImage"] img {
        border-radius: 16px;
        border: 2px solid rgba(255,255,255,0.8);
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.15);
    }

    /* Métriques modernes */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--gray-100);
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
    }

    div[data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--primary), var(--accent));
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--gray-500);
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetricValue"] {
        color: var(--dark);
        font-weight: 800;
        font-size: 1.5rem;
    }

    /* Résultats */
    .result-success {
        background: linear-gradient(135deg, var(--success-light), white);
        border: 2px solid rgba(16, 185, 129, 0.2);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: slideUp 0.5s ease-out;
    }

    .result-danger {
        background: linear-gradient(135deg, var(--danger-light), white);
        border: 2px solid rgba(239, 68, 68, 0.2);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: slideUp 0.5s ease-out;
    }

    .result-icon {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem;
        font-size: 2rem;
    }

    .result-success .result-icon {
        background: var(--success-light);
        color: var(--success);
        animation: float 3s ease-in-out infinite;
    }

    .result-danger .result-icon {
        background: var(--danger-light);
        color: var(--danger);
        animation: float 3s ease-in-out infinite;
    }

    .result-title {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .result-success .result-title {
        color: var(--success);
    }

    .result-danger .result-title {
        color: var(--danger);
    }

    /* Progress bar custom */
    .custom-progress {
        width: 100%;
        height: 12px;
        background: var(--gray-100);
        border-radius: 999px;
        overflow: hidden;
        margin-top: 1rem;
    }

    .custom-progress-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        transition: width 0.8s ease-out;
    }

    /* Footer */
    .modern-footer {
        background: linear-gradient(135deg, var(--dark), var(--gray-900));
        border-radius: 20px;
        padding: 2rem;
        color: white;
        margin-top: 2rem;
    }

    .modern-footer h3 {
        color: white;
        font-weight: 700;
    }

    .modern-footer p {
        color: #94a3b8;
        line-height: 1.7;
    }

    /* Placeholder */
    .placeholder-container {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border: 2px dashed #e2e8f0;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
    }

    .placeholder-icon {
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, var(--primary-light), #dbeafe);
        border-radius: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.5rem;
        font-size: 2.5rem;
    }

    /* Typo */
    h1, h2, h3 {
        color: var(--dark);
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    p, .small-note {
        color: var(--gray-500);
        line-height: 1.7;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--gray-100);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--gray-300);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--gray-500);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOGO - Design moderne avec gradient
# ============================================================
def create_modern_logo(size=96):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Fond avec gradient simulé
    for i in range(size):
        ratio = i / size
        r = int(14 + (99 - 14) * ratio)
        g = int(165 + (102 - 165) * ratio)
        b = int(233 + (241 - 233) * ratio)
        draw.line([(0, i), (size, i)], fill=(r, g, b, 255))

    # Cercle intérieur
    padding = 8
    draw.rounded_rectangle(
        [padding, padding, size - padding, size - padding],
        radius=20,
        fill=(255, 255, 255, 255)
    )

    # Croix médicale stylisée
    cx, cy = size // 2, size // 2
    bar_w, bar_h = 10, 36
    draw.rounded_rectangle(
        [cx - bar_w // 2, cy - bar_h // 2, cx + bar_w // 2, cy + bar_h // 2],
        radius=5,
        fill=(14, 165, 233, 255)
    )
    draw.rounded_rectangle(
        [cx - bar_h // 2, cy - bar_w // 2, cx + bar_h // 2, cy + bar_w // 2],
        radius=5,
        fill=(14, 165, 233, 255)
    )

    return img


def create_placeholder_brain(size=120):
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Cercle extérieur
    draw.ellipse(
        [4, 4, size - 4, size - 4],
        outline=(203, 213, 225, 255),
        width=3
    )

    # Lignes de cerveau stylisées
    cx, cy = size // 2, size // 2
    # Ligne médiane
    draw.line([(cx, 20), (cx, size - 20)], fill=(148, 163, 184, 255), width=2)
    # Courbes
    draw.arc([cx - 25, cy - 20, cx + 25, cy + 20], 270, 90, fill=(148, 163, 184, 255), width=2)
    draw.arc([cx - 20, cy - 15, cx + 20, cy + 15], 90, 270, fill=(148, 163, 184, 255), width=2)

    return img


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    model = None
    st.error(f"Erreur de chargement du modèle : {e}")


# ============================================================
# IMAGE PROCESSING
# ============================================================
def preprocess_image(pil_img):
    img = pil_img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = keras_image.img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return img, arr


def make_gradcam(model, img_array):
    try:
        conv_layers = [
            layer for layer in model.layers
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))
        ]
        if len(conv_layers) == 0:
            return None
        last_conv_layer = conv_layers[-1]
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = np.maximum(heatmap.numpy(), 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        return heatmap
    except Exception:
        return None


def create_brain_mask(pil_img):
    img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)).convert("L"))
    _, mask = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    brain_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.drawContours(brain_mask, [largest], -1, 1, thickness=-1)
    erosion_kernel = np.ones((9, 9), np.uint8)
    brain_mask = cv2.erode(brain_mask, erosion_kernel, iterations=1)
    return brain_mask


def create_heatmap_overlay(pil_img, heatmap):
    img_pil = pil_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    img = np.array(img_pil)
    brain_mask = create_brain_mask(img_pil)
    masked_heatmap = heatmap * brain_mask
    if masked_heatmap.max() > 1e-8:
        masked_heatmap = masked_heatmap / masked_heatmap.max()
    heatmap_uint8 = np.uint8(255 * masked_heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.65, heatmap_color, 0.35, 0)
    return Image.fromarray(overlay)


def show_centered_image(img, caption="", width=DISPLAY_WIDTH):
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.image(img, caption=caption, width=width)


# ============================================================
# SIDEBAR MODERNE
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="display: inline-block; position: relative;">
                <div style="position: absolute; inset: -4px; background: linear-gradient(135deg, #0ea5e9, #6366f1); border-radius: 20px; opacity: 0.3; filter: blur(8px);"></div>
        """,
        unsafe_allow_html=True
    )
    st.image(create_modern_logo(80), width=64)
    st.markdown(
        """
            </div>
            <h2 style="margin-top: 1rem; font-size: 1.3rem;">MediScan AI</h2>
            <p style="color: #64748b; font-size: 0.8rem;">Diagnostic Assisté par IA</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### ⚙️ Paramètres d'analyse")

    threshold = st.slider(
        "Seuil de décision",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Ajustez la sensibilité de détection"
    )

    show_heatmap = st.toggle(
        "Carte Grad-CAM",
        value=True,
        help="Visualise les zones d'attention du modèle"
    )

    st.divider()

    st.markdown("### 📊 Informations")
    st.caption("🧠 Modèle : DenseNet121")
    st.caption("📐 Entrée : 224 × 224 px")
    st.caption("🔬 Tâche : Classification binaire")
    st.caption("🎓 Usage : Académique")

    st.divider()

    st.markdown(
        """
        <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem;">
            <p style="color: #94a3b8; font-size: 0.75rem; margin: 0;">
                Cette interface est destinée à un usage académique uniquement.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER MODERNE
# ============================================================
header_left, header_right = st.columns([0.11, 0.89], gap="medium")

with header_left:
    st.image(create_modern_logo(100), width=72)

with header_right:
    st.markdown(
        """
        <div class="aero-header">
            <div class="header-badge">
                <span class="dot"></span>
                Système opérationnel
            </div>
            <div class="header-title">MediScan AI</div>
            <div class="header-subtitle">
                Plateforme intelligente d'analyse d'images IRM cérébrales par deep learning.
                Classification binaire avec visualisation explicable (Grad-CAM) pour 
                l'assistance au diagnostic médical.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Alerte moderne
st.markdown(
    """
    <div class="modern-alert">
        <div class="modern-alert-icon">⚠️</div>
        <div class="modern-alert-text">
            <strong>Usage académique uniquement</strong> — Cette application est un prototype 
            de recherche. Elle ne remplace en aucun cas l'expertise d'un professionnel de santé.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MAIN LAYOUT
# ============================================================
left, right = st.columns([0.95, 1.05], gap="large")

with left:
    with st.container(border=False):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📤 Import d'image")

        uploaded_file = st.file_uploader(
            "Glissez-déposez ou cliquez pour sélectionner",
            type=["jpg", "jpeg", "png"],
            help="Formats acceptés : JPG, JPEG, PNG"
        )

        st.caption(
            "💡 **Conseil** : Utilisez des images IRM natives sans annotations, "
            "watermarks ou flèches pour des résultats optimaux."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        btn_col_1, btn_col_2 = st.columns([0.55, 0.45])
        with btn_col_1:
            analyze = st.button("🔍 Lancer l'analyse", type="primary", use_container_width=True)
        with btn_col_2:
            reset_view = st.button("🔄 Réinitialiser", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

with right:
    with st.container(border=False):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🖼️ Aperçu")

        if uploaded_file is not None:
            original_img = Image.open(uploaded_file).convert("RGB")
            show_centered_image(original_img, caption="Image importée", width=DISPLAY_WIDTH)
        else:
            ph_col_1, ph_col_2, ph_col_3 = st.columns([1, 1.5, 1])
            with ph_col_2:
                st.markdown(
                    """
                    <div class="placeholder-container">
                        <div class="placeholder-icon">🧠</div>
                        <h4 style="color: #1e293b; margin-bottom: 0.5rem;">Aucune image</h4>
                        <p class="small-note">Importez une image IRM pour commencer l'analyse</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# ANALYSIS
# ============================================================
if uploaded_file is not None and analyze:
    if model is None:
        st.error(
            "❌ Le modèle n'a pas pu être chargé. Vérifiez que le fichier "
            "`densenet_brain_tumor.h5` est présent dans le répertoire de l'application."
        )
    else:
        with st.spinner("🧠 Analyse en cours par le réseau de neurones..."):
            start_time = time.time()
            processed_img, img_array = preprocess_image(original_img)
            probability = float(model.predict(img_array, verbose=0)[0][0])
            heatmap = make_gradcam(model, img_array)
            inference_time = (time.time() - start_time) * 1000

        st.markdown("<br>", unsafe_allow_html=True)

        # Résultat moderne
        is_tumor = probability >= threshold
        percent = probability * 100

        if is_tumor:
            st.markdown(
                f"""
                <div class="result-danger">
                    <div class="result-icon">🔴</div>
                    <div class="result-title">Tumeur détectée</div>
                    <p style="color: #64748b; margin-bottom: 1rem;">
                        Le modèle identifie une probabilité élevée de présence tumorale.
                    </p>
                    <div class="custom-progress">
                        <div class="custom-progress-fill" style="width: {percent}%; background: linear-gradient(90deg, #ef4444, #f87171);"></div>
                    </div>
                    <p style="color: #ef4444; font-weight: 700; margin-top: 0.5rem;">{percent:.1f}% de confiance</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-success">
                    <div class="result-icon">🟢</div>
                    <div class="result-title">Aucune tumeur détectée</div>
                    <p style="color: #64748b; margin-bottom: 1rem;">
                        Le modèle indique une faible probabilité de présence tumorale.
                    </p>
                    <div class="custom-progress">
                        <div class="custom-progress-fill" style="width: {percent}%; background: linear-gradient(90deg, #10b981, #34d399);"></div>
                    </div>
                    <p style="color: #10b981; font-weight: 700; margin-top: 0.5rem;">{percent:.1f}% de confiance</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Métriques en ligne
        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Probabilité", f"{percent:.2f}%", delta="tumeur" if is_tumor else "sain")
        metric_2.metric("Seuil", f"{threshold:.2f}")
        metric_3.metric("Temps", f"{inference_time:.0f} ms", delta="rapide")

        st.markdown("<br>", unsafe_allow_html=True)

        if heatmap is not None:
            analysis_col_1, analysis_col_2 = st.columns(2, gap="large")

            with analysis_col_1:
                with st.container(border=False):
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("🔬 Image prétraitée")
                    show_centered_image(
                        processed_img,
                        caption="Redimensionnée 224×224",
                        width=DISPLAY_WIDTH
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

            with analysis_col_2:
                with st.container(border=False):
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("🎯 Carte d'attention")

                    if show_heatmap:
                        overlay = create_heatmap_overlay(processed_img, heatmap)
                        show_centered_image(
                            overlay,
                            caption="Grad-CAM — Zones d'intérêt",
                            width=DISPLAY_WIDTH
                        )
                        st.caption(
                            "🔍 **Grad-CAM** : Visualisation des zones qui influencent "
                            "la décision du modèle. Ne constitue pas une segmentation clinique."
                        )
                    else:
                        st.info("Visualisation désactivée dans les paramètres.")

                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning(
                "⚠️ Grad-CAM indisponible pour cette architecture. "
                "La classification fonctionne normalement."
            )


# ============================================================
# FOOTER MODERNE
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="modern-footer">
        <h3>🧠 À propos de MediScan AI</h3>
        <p>
            Système académique de classification binaire des images IRM cérébrales 
            utilisant un réseau DenseNet121 pré-entraîné. La carte Grad-CAM fournit 
            une interprétabilité visuelle des décisions algorithmiques.
        </p>
        <p style="margin-top: 1rem; font-size: 0.85rem; color: #475569;">
            ⚠️ Cet outil est strictement réservé à la recherche et à l'enseignement. 
            Il ne doit jamais être utilisé pour un diagnostic clinique réel.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)