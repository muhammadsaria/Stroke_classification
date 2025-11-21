"""
Full Streamlit app: Classification + XAI (GradCAM, IG, LIME) + Multi-scan Compare + Doctor CSV + Advanced Drift Dashboard
Model path used: "best_monai_densenet121.pth"

Instructions:
- Put this file in the project folder where your "best_monai_densenet121.pth" exists (or update MODEL_PATH below).
- Recommended (if you still see Streamlit watcher issues): run
    streamlit run app.py --server.runOnSave=false
"""

# -------------------------
# ENV & WARNINGS (suppress noisy logs)
# -------------------------
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"                # hide oneDNN info
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"     # attempt to reduce watcher reloads

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------
# IMPORTS
# -------------------------
import io
import tempfile
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go

# Torch/Monai/Explainability libs
import torch
# defensive monkeypatch for streamlit watcher + torch._classes on some environments (avoids the '__path__._path' error)
try:
    if hasattr(torch, "_classes"):
        try:
            # ensure torch._classes has a harmless __path__ to stop Streamlit from trying to access .__path__._path
            setattr(torch._classes, "__path__", [])
        except Exception:
            pass
except Exception:
    pass

import torch.nn.functional as F
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity
from monai.networks.nets import DenseNet121

# Captum & LIME
from captum.attr import IntegratedGradients
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Drift stats
from scipy.stats import entropy, ks_2samp
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# APP CONFIG
# -------------------------
st.set_page_config(page_title="🧠 Stroke Classifier + XAI + Drift", layout="wide", page_icon="🧠")

MODEL_PATH = "best_monai_densenet121.pth"   # <- update if your model is elsewhere
LOGS_CSV = "predictions_log.csv"
CLASSES = ["Normal", "Stroke"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
single_transform = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

# Dark professional styling
st.markdown(
    """
    <style>
    body { background-color: #0d0f11; color: #e6f7eb; }
    .stButton>button { background-color:#151617; color:#e6f7eb; border:1px solid #1f7a5a; }
    .stDownloadButton>button { background-color:#151617; color:#e6f7eb; border:1px solid #1f7a5a; }
    .stFileUploader>div { color:#e6f7eb; }
    h1,h2,h3,h4 { color:#7ef8b3; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Automated Brain Stroke Classification — XAI + Drift Dashboard")

# -------------------------
# HELPERS
# -------------------------
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

def safe_torch_load(path, map_location=DEVICE):
    """
    Try to load state dict safely with weights_only=True when supported.
    Fallback to plain torch.load if weights_only not available.
    """
    try:
        # preferred (PyTorch newer APIs)
        state = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # older torch doesn't support weights_only keyword
        state = torch.load(path, map_location=map_location)
    return state

def load_model(path=MODEL_PATH):
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(DEVICE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at '{path}'. Put your 'best_monai_densenet121.pth' there or update MODEL_PATH.")
    state = safe_torch_load(path)
    # handle nested state_dicts
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        # strip module. prefix if present
        state = { (k.replace("module.", "") if isinstance(k, str) else k): v for k, v in state.items() }
        model.load_state_dict(state)
    else:
        # possibly the entire model object was saved
        model = state
    model.eval()
    return model

def predict_with_monai(path, model):
    # monai transform returns numpy or tensor; convert carefully
    img = single_transform(path)
    # ensure numpy array -> torch tensor safely
    if isinstance(img, torch.Tensor):
        tensor = img.clone().detach().unsqueeze(0).to(DEVICE)
    else:
        arr = np.array(img, dtype=np.float32)
        tensor = torch.from_numpy(arr.copy()).float().unsqueeze(0).to(DEVICE)  # shape (1,C,H,W)
    with torch.no_grad():
        out = model(tensor)
        probs = F.softmax(out, dim=1)
        conf, idx = torch.max(probs, dim=1)
    return CLASSES[int(idx.item())], float(conf.item()*100.0), probs.cpu().numpy()[0]

def append_log(entry: dict):
    df = pd.DataFrame([entry])
    if os.path.exists(LOGS_CSV):
        df.to_csv(LOGS_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(LOGS_CSV, index=False)

def load_logs_df():
    if os.path.exists(LOGS_CSV):
        return pd.read_csv(LOGS_CSV)
    return pd.DataFrame(columns=["timestamp","first_name","last_name","age","gender","image_name","prediction","confidence"])

def find_last_conv_layer(model):
    # return last nn.Conv2d layer
    import torch.nn as nn
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise RuntimeError("No Conv2d layers found in model.")
    return convs[-1]

# -------------------------
# XAI utils
# -------------------------
def preprocess_image_to_tensor_for_xai(img_path):
    # simple 1-channel 224x224, returns (H,W) uint8 and tensor (1,1,H,W)
    pil = Image.open(img_path).convert("L").resize((224,224))
    np_img = np.array(pil).astype(np.float32)
    t = torch.from_numpy(np_img.copy()).float().unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,H,W)
    return np_img, t

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()
    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            # grad_out is tuple; take first
            self.gradients = grad_out[0].detach()
        try:
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_full_backward_hook(backward_hook)
        except Exception:
            # fallback for older pytorch
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_backward_hook(backward_hook)
    def generate_cam(self, input_tensor, class_idx):
        # input_tensor shape (1,1,H,W)
        logits = self.model(input_tensor)
        score = logits[0, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        grads = self.gradients            # (B,C,h,w)
        acts = self.activations          # (B,C,h,w)
        weights = grads.mean(dim=(2,3), keepdim=True)   # (B,C,1,1)
        cam = (weights * acts).sum(dim=1).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
        return cam

def integrated_gradients_explain(model, input_tensor, target_index, steps=50):
    ig = IntegratedGradients(model)
    attributions = ig.attribute(input_tensor, target=target_index, n_steps=steps)
    attr = attributions.detach().cpu().numpy().squeeze()
    if attr.ndim == 3 and attr.shape[0] == 1:
        attr = attr[0]
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-9)
    return attr

def lime_explain_img(img_rgb_uint8, model, samples=300):
    explainer = lime_image.LimeImageExplainer()
    def batch_predict(images_np):
        # images_np: list of HxWx3 uint8
        imgs = []
        for im in images_np:
            im_f = im.astype(np.float32) / 255.0
            gray = np.mean(im_f, axis=2, keepdims=True)   # HxWx1
            t = torch.from_numpy(gray.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)
            imgs.append(t)
        batch = torch.cat(imgs, dim=0)
        with torch.no_grad():
            out = model(batch)
            probs = F.softmax(out, dim=1).cpu().numpy()
        return probs
    exp = explainer.explain_instance(img_rgb_uint8, batch_predict, top_labels=2, hide_color=0, num_samples=samples)
    return exp

# -------------------------
# Drift functions & dashboard (enhanced)
# -------------------------
def get_embeddings(model, img_path):
    img = single_transform(img_path)
    if isinstance(img, torch.Tensor):
        t = img.clone().detach().unsqueeze(0).to(DEVICE)
    else:
        t = torch.from_numpy(np.array(img, dtype=np.float32).copy()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = model.features(t)                       # B x C x h x w
        pooled = F.adaptive_avg_pool2d(feats, (1,1))
        vec = pooled.view(pooled.size(0), -1)           # B x D
    return vec.cpu().numpy()

def compute_drift_metrics_enhanced(baseline_emb, incoming_emb):
    # baseline_emb, incoming_emb: 2D arrays (N, D) or 1D flattened
    b = np.array(baseline_emb).flatten()
    i = np.array(incoming_emb).flatten()
    # trim or pad for numerical stability
    n = min(len(b), len(i), 100000)
    b = b[:n]; i = i[:n]
    # cosine:
    cos_sim = (np.dot(b, i) / (np.linalg.norm(b) * np.linalg.norm(i) + 1e-12)).item()
    # hist-based KL (using hist of flattened values)
    bh, _ = np.histogram(b, bins=50, density=True)
    ih, _ = np.histogram(i, bins=50, density=True)
    kl = float(entropy(bh + 1e-10, ih + 1e-10))
    # KS statistic
    ks_stat, ks_p = ks_2samp(b, i)
    # psi (population stability index)
    def psi(a,bins=50):
        ah, bins = np.histogram(b, bins=bins)
        bh, _ = np.histogram(a, bins=bins)
        a_pct = bh / (bh.sum() + 1e-9)
        b_pct = ah / (ah.sum() + 1e-9)
        return float(np.sum((a_pct - b_pct) * np.log((a_pct + 1e-9)/(b_pct + 1e-9))))
    try:
        psi_val = psi(b, bins=50)
    except Exception:
        psi_val = 0.0
    # Decision: thresholds (fine-tune as needed)
    drift = (cos_sim < 0.90) or (kl > 0.5) or (ks_stat > 0.3) or (psi_val > 0.1)
    return {"cosine_similarity": cos_sim, "kl_divergence": kl, "ks_stat": ks_stat, "psi": psi_val, "drift": bool(drift)}

def show_advanced_drift_dashboard(baseline_emb, incoming_emb):
    metrics = compute_drift_metrics_enhanced(baseline_emb, incoming_emb)
    status = "DRIFT DETECTED" if metrics["drift"] else "No significant drift"
    st.subheader("Advanced Drift Monitoring")
    if metrics["drift"]:
        st.error(f"🚨 {status} — Cosine={metrics['cosine_similarity']:.3f} KL={metrics['kl_divergence']:.3f} KS={metrics['ks_stat']:.3f} PSI={metrics['psi']:.3f}")
    else:
        st.success(f"✅ {status} — Cosine={metrics['cosine_similarity']:.3f} KL={metrics['kl_divergence']:.3f} KS={metrics['ks_stat']:.3f} PSI={metrics['psi']:.3f}")
    # visuals similar to the code you provided, but plotted with streamlit
    fig = go.Figure(go.Indicator(mode="gauge+number", value=metrics["cosine_similarity"],
                                title={"text":"Cosine Similarity"}, gauge={"axis":{"range":[0,1]}}))
    st.plotly_chart(fig, use_container_width=True)

    # Matplotlib visuals in a grid
    fig, axs = plt.subplots(3,2, figsize=(14,12))
    bflat = np.array(baseline_emb).flatten()[:8000]
    if bflat.size == 0:
        bflat = np.zeros(100)
    if isinstance(incoming_emb, np.ndarray) or isinstance(incoming_emb, list):
        if np.array(incoming_emb).size == 0:
            inflat = np.zeros_like(bflat)
        else:
            inflat = np.array(incoming_emb).flatten()[:8000]
    else:
        inflat = np.zeros_like(bflat)

    sns.kdeplot(bflat, ax=axs[0,0], label="Baseline")
    sns.kdeplot(inflat, ax=axs[0,0], label="Incoming")
    axs[0,0].set_title("KDE Comparison")

    axs[0,1].hist(bflat, bins=50, alpha=0.5, label="Baseline")
    axs[0,1].hist(inflat, bins=50, alpha=0.5, label="Incoming")
    axs[0,1].set_title("Histogram Comparison")

    axs[1,0].boxplot([bflat[:5000], inflat[:5000]], labels=["Baseline","Incoming"])
    axs[1,0].set_title("Boxplot")

    diff = inflat[:5000] - bflat[:5000]
    sns.kdeplot(diff, ax=axs[1,1], color='black')
    axs[1,1].set_title("Difference KDE")

    axs[2,0].bar(["KL"], [metrics["kl_divergence"]], color='purple')
    axs[2,0].axhline(0.5, color='red', linestyle='--')

    axs[2,1].bar(["KS"], [metrics["ks_stat"]], color='blue')
    axs[2,1].axhline(0.3, color='red', linestyle='--')

    plt.tight_layout()
    st.pyplot(fig)

    # CSV download
    df = pd.DataFrame([metrics])
    st.download_button("Download drift metrics (CSV)", df.to_csv(index=False).encode(), "drift_metrics.csv", "text/csv")
    return metrics

# -------------------------
# Load model (safe)
# -------------------------
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("Model load failed: " + str(e))
    st.stop()

# -------------------------
# UI: sidebar navigation
# -------------------------
page = st.sidebar.selectbox("Page", ["Classification", "XAI", "Multi-scan Compare", "Doctor Summary", "Drift Analysis"])

# -------------------------
# PAGE: Classification
# -------------------------
if page == "Classification":
    st.header("Classification")
    col1, col2 = st.columns(2)
    first_name = col1.text_input("First name")
    last_name = col2.text_input("Last name")
    col3, col4 = st.columns(2)
    age = col3.number_input("Age", min_value=0, max_value=120, value=30)
    gender = col4.selectbox("Gender", ["Male", "Female", "Other"])

    uploaded = st.file_uploader("Upload CT image (png/jpg/jpeg)", type=["png","jpg","jpeg"])
    if uploaded:
        path = save_uploaded_file(uploaded)
        st.image(path, caption="Uploaded CT", use_container_width=True)
        if st.button("Predict"):
            pred, conf, probs = predict_with_monai(path, model)
            st.success(f"Prediction: {pred}  —  Confidence: {conf:.2f}%")
            append_log({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "first_name": first_name, "last_name": last_name, "age": age, "gender": gender,
                "image_name": os.path.basename(path), "prediction": pred, "confidence": round(conf,2)
            })
            if pred == "Stroke":
                st.warning("""
                ⚠️ **Immediate precautions (example)**:
                - Call emergency services; time is critical.
                - Do not give food or drink if consciousness altered.
                - Keep patient supine and monitor breathing.
                - Take note of symptom onset time and notify clinicians.
                """)

# -------------------------
# PAGE: XAI
# -------------------------
elif page == "XAI":
    st.header("Explainable AI — IG + GradCAM + LIME")
    uploaded = st.file_uploader("Upload CT for explanation", type=["png","jpg","jpeg"])
    if uploaded:
        path = save_uploaded_file(uploaded)
        st.image(path, caption="Input CT", use_container_width=True)
        if st.button("Run XAI"):
            img_np, tensor = preprocess_image_to_tensor_for_xai(path)   # (H,W), (1,1,H,W)
            # predict class
            with torch.no_grad():
                out = model(tensor)
                pred_idx = int(torch.argmax(out, dim=1).item())
            # Integrated Gradients
            ig_map = integrated_gradients_explain(model, tensor, pred_idx, steps=40)
            # GradCAM (find last conv)
            last_conv = find_last_conv_layer(model)
            gradcam_gen = GradCAM(model, last_conv)
            cam = gradcam_gen.generate_cam(tensor, pred_idx)
            # LIME
            rgb_vis = np.stack([img_np]*3, axis=-1).astype(np.uint8)
            lime_exp = lime_explain_img(rgb_vis, model, samples=300)
            top_label = lime_exp.top_labels[0]
            temp, mask = lime_exp.get_image_and_mask(top_label, positive_only=True, num_features=5, hide_rest=False)

            # Show side-by-side
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("Integrated Gradients")
                st.image((ig_map * 255).astype(np.uint8), use_container_width=True)
            with c2:
                st.subheader("GradCAM Overlay")
                # create overlay: combine grayscale and cam
                cam_resized = np.uint8(plt.cm.jet(cam)[..., :3] * 255)
                base_rgb = np.stack([img_np]*3, axis=-1).astype(np.uint8)
                overlay = cv2.addWeighted(base_rgb, 0.6, cam_resized, 0.4, 0)
                st.image(overlay, use_container_width=True)
            with c3:
                st.subheader("LIME")
                st.image(mark_boundaries(temp, mask), use_container_width=True)

# -------------------------
# PAGE: Multi-scan Compare
# -------------------------
elif page == "Multi-scan Compare":
    st.header("Multi-scan Comparison (Batch predictions + CSV)")
    multi = st.file_uploader("Upload multiple CTs (select many)", type=["png","jpg","jpeg"], accept_multiple_files=True)
    if multi and st.button("Run Batch"):
        rows = []
        for up in multi:
            p = save_uploaded_file(up)
            pred, conf, _ = predict_with_monai(p, model)
            rows.append({"image": up.name, "prediction": pred, "confidence": round(conf,2)})
        df = pd.DataFrame(rows)
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False).encode(), "multi_scan_report.csv", "text/csv")

# -------------------------
# PAGE: Doctor Summary
# -------------------------
elif page == "Doctor Summary":
    st.header("Doctor Summary & Logs")
    df = load_logs_df()
    if df.empty:
        st.info("No logs yet.")
    else:
        st.dataframe(df.tail(200))
        st.download_button("Download logs CSV", df.to_csv(index=False).encode(), "doctor_logs.csv", "text/csv")

# -------------------------
# PAGE: Drift Analysis
# -------------------------
elif page == "Drift Analysis":
    st.header("Drift Analysis — Upload baseline (zip or files) & incoming scans")
    st.markdown("**Option A:** upload baseline images (many) and incoming images; **Option B:** provide local baseline folder path (server only).")

    baseline_files = st.file_uploader("Upload baseline images (multiple)", type=["png","jpg","jpeg"], accept_multiple_files=True, key="baseline")
    incoming_files = st.file_uploader("Upload incoming images (multiple)", type=["png","jpg","jpeg"], accept_multiple_files=True, key="incoming")
    baseline_folder = st.text_input("Or (server) baseline folder path (optional)", value="")

    if st.button("Run Drift"):
        # prepare baseline embeddings
        baseline_embs = []
        if baseline_files:
            for f in baseline_files:
                p = save_uploaded_file(f)
                baseline_embs.append(get_embeddings(model, p))
        elif baseline_folder and os.path.exists(baseline_folder):
            # expect subfolders or images directly
            for fn in os.listdir(baseline_folder):
                if fn.lower().endswith((".png",".jpg",".jpeg")):
                    baseline_embs.append(get_embeddings(model, os.path.join(baseline_folder, fn)))
        else:
            st.error("Provide baseline images or a valid baseline folder.")
            baseline_embs = []

        incoming_embs = []
        if incoming_files:
            for f in incoming_files:
                p = save_uploaded_file(f)
                incoming_embs.append(get_embeddings(model, p))
        else:
            st.error("Upload incoming images.")
            incoming_embs = []

        if len(baseline_embs) == 0 or len(incoming_embs) == 0:
            st.warning("Need at least one baseline and one incoming embedding to run drift.")
        else:
            baseline_emb_arr = np.vstack(baseline_embs)
            incoming_emb_arr = np.vstack(incoming_embs)
            metrics = show_advanced_drift_dashboard(baseline_emb_arr, incoming_emb_arr)  # visual & csv
            st.write("Drift metrics (returned):", metrics)

# -------------------------
# end
# -------------------------

