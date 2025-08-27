# -*- coding: utf-8 -*-
"""Faraz Streamlit app – batch reviews input (v2)

Changes vs your version:
- Accept multiple reviews (one per line OR a Python/JSON list)
- Batch featurization & prediction for speed
- Results table with per‑review stars + expected rating
- Aggregate summary + CSV download
"""

# !pip install sentence-transformers xgboost scikit-learn pandas protobuf==3.20.* streamlit textblob

import os, ast, io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ================== CONFIG ==================
ARTIFACT_DIR   = "artifacts"
META_PATH      = f"{ARTIFACT_DIR}/metadata.joblib"
XGB_PATH       = f"{ARTIFACT_DIR}/xgb_model.joblib"
LR_PATH        = f"{ARTIFACT_DIR}/logreg_sent.joblib"
SCALER_PATH    = f"{ARTIFACT_DIR}/sent_scaler.joblib"
META_CLF_PATH  = f"{ARTIFACT_DIR}/meta_model.joblib"

# ================== STREAMLIT PAGE ==================
st.set_page_config(page_title="Product Quality Predictor", layout="wide")
st.title("Product Quality Predictor")
st.subheader("Check if your product is up to standard")
st.caption("Paste multiple qualitative reviews – one per line – or a Python/JSON list. We'll predict a star rating for each.")

# ================== LOAD ARTIFACTS (cached) ==================
@st.cache_resource
def load_artifacts():
    required = [META_PATH, XGB_PATH, LR_PATH, SCALER_PATH, META_CLF_PATH]
    if not all(os.path.exists(p) for p in required):
        st.error(
            "Model artifacts not found. Please ensure the following files exist in 'artifacts/':\n"
            "- metadata.joblib\n- xgb_model.joblib\n- logreg_sent.joblib\n- sent_scaler.joblib\n- meta_model.joblib\n"
            "(If you need a training script, re‑enable your original training block.)"
        )
        st.stop()

    metadata = joblib.load(META_PATH)
    embedder = SentenceTransformer(metadata.get("embed_model_name", "all-MiniLM-L12-v2"))
    xgb      = joblib.load(XGB_PATH)
    logreg   = joblib.load(LR_PATH)
    scaler   = joblib.load(SCALER_PATH)
    meta_clf = joblib.load(META_CLF_PATH)
    return embedder, scaler, xgb, logreg, meta_clf, metadata

embedder, scaler, xgb, logreg, meta_clf, metadata = load_artifacts()

# ================== HELPERS ==================
def parse_reviews(text: str):
    """Accepts either newline‑separated text or a Python/JSON list and returns a list of non‑empty strings."""
    text = (text or "").strip()
    if not text:
        return []
    # Try to parse as a Python/JSON list first
    reviews = []
    if text.startswith("["):
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, (list, tuple)):
                reviews = [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            reviews = []
    # Fallback: treat as newline‑separated
    if not reviews:
        reviews = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return reviews


def featurize_batch(reviews):
    """Return embeddings and scaled sentiment features for a list of review strings."""
    if len(reviews) == 0:
        return np.empty((0, 384)), np.empty((0, 2))

    # Embeddings (MiniLM variants commonly output 384 dims)
    Xe = embedder.encode(reviews, show_progress_bar=False, batch_size=min(64, max(1, len(reviews))))

    # Sentiment
    pol_sub = []
    for r in reviews:
        try:
            b = TextBlob(str(r))
            pol_sub.append((float(b.sentiment.polarity), float(b.sentiment.subjectivity)))
        except Exception:
            pol_sub.append((0.0, 0.0))
    Xs = np.asarray(pol_sub, dtype=float)
    Xs_sc = scaler.transform(Xs)
    return Xe, Xs_sc


def predict_stars_batch(reviews):
    """Predict discrete stars (1–5) and probabilities for each review in a batch.
    Returns (stars: (n,), probs: (n,5))."""
    Xe, Xs_sc = featurize_batch(reviews)
    if Xe.shape[0] == 0:
        return np.array([]), np.empty((0, 5))

    p_xgb = xgb.predict_proba(Xe)        # (n,5)
    p_lr  = logreg.predict_proba(Xs_sc)  # (n,5)
    Z     = np.hstack([p_xgb, p_lr])     # (n,10)
    probs = meta_clf.predict_proba(Z)    # (n,5)
    stars = np.argmax(probs, axis=1) + 1
    return stars.astype(int), probs


# ================== UI ==================
with st.expander("⚙️ Input Parameters", expanded=True):
    default_text = (
        "Waistband stayed up during squats; fabric felt durable.\n"
        "Loved the fit but stitching came loose after two washes.\n"
        "Material is soft, didn’t ride down, and sweat‑wicking was solid."
    )
    bulk_input = st.text_area(
        "Paste reviews (one per line) OR a Python/JSON list:",
        value=default_text,
        height=180,
    )

cols = st.columns([1, 1, 2])
with cols[0]:
    run_infer = st.button("🚀 Predict for All Reviews")

if run_infer:
    reviews = parse_reviews(bulk_input)
    if not reviews:
        st.warning("Please paste at least one non‑empty review.")
        st.stop()

    stars, probs = predict_stars_batch(reviews)
    exp_rating = (probs * np.arange(1, 6, dtype=float)).sum(axis=1)  # expected stars per review

    # Results table
    df_out = pd.DataFrame({
        "review": reviews,
        "pred_stars": stars,
        "expected_stars": np.round(exp_rating, 2),
    })

    # Aggregate summary
    mean_expected = float(np.mean(exp_rating)) if len(exp_rating) else float("nan")
    star_counts = pd.Series(stars).value_counts().sort_index()

    s1, s2, s3 = st.columns([1, 1, 2])
    with s1:
        st.metric("Mean expected rating", f"{mean_expected:.2f}★")
    with s2:
        st.write("**Predicted distribution**")
        st.write({f"{k}★": int(v) for k, v in star_counts.items()})

    st.dataframe(df_out, use_container_width=True, hide_index=True)

    # Download
    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download predictions as CSV",
        data=csv_bytes,
        file_name="review_predictions.csv",
        mime="text/csv",
    )

st.caption("Artifacts are cached in memory via `@st.cache_resource`. Paste multiple reviews to score them in one go.")
