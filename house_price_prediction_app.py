import streamlit as st
import os
import io
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght=700&family=DM+Sans:wght=400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hero */
.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem; color: #f9d77e; margin-bottom: 0.3rem;
    text-shadow: 0 2px 20px rgba(249,215,126,0.35);
}
.hero p { font-size: 1.05rem; color: #b8b8d1; max-width: 520px; margin: 0 auto; line-height: 1.6; }

/* Card Wrap */
.card-container {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px; padding: 2rem 2.2rem;
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
}

/* Section label */
.section-label {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #f9d77e; margin-bottom: 1rem;
}

/* Widget overrides */
label { color: #d4d4e8 !important; font-size: 0.9rem !important; font-weight: 500 !important; }
.stNumberInput input, .stSlider { background: rgba(255,255,255,0.08) !important; border-radius: 10px !important; color: #fff !important; }
div[data-baseweb="input"] { border-radius: 10px !important; }
div[data-baseweb="input"] input { color: #fff !important; background: rgba(255,255,255,0.08) !important; }

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #f9d77e, #f4a261);
    color: #1a1a2e; font-family: 'DM Sans', sans-serif;
    font-weight: 700; font-size: 1.05rem; letter-spacing: 0.04em;
    padding: 0.75rem 0; border: none; border-radius: 12px;
    cursor: pointer; transition: opacity 0.2s, transform 0.15s;
    box-shadow: 0 4px 24px rgba(249,215,126,0.3);
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* Result box */
.result-box {
    text-align: center;
    background: linear-gradient(135deg, rgba(249,215,126,0.12), rgba(244,162,97,0.12));
    border: 1px solid rgba(249,215,126,0.35);
    border-radius: 18px; padding: 2rem; margin-top: 1.5rem;
}
.result-box .label {
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #f9d77e; margin-bottom: 0.5rem;
}
.result-box .price {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem; color: #ffffff; margin: 0.2rem 0;
    text-shadow: 0 2px 16px rgba(255,255,255,0.15);
}
.result-box .summary { font-size: 0.88rem; color: #9898b8; margin-top: 0.6rem; }

/* Error box */
.error-box {
    background: rgba(255,80,80,0.12); border: 1px solid rgba(255,100,100,0.4);
    border-radius: 12px; padding: 1rem 1.2rem; color: #ff9090;
    font-size: 0.9rem; margin-top: 1rem;
}

/* Info box */
.info-box {
    background: rgba(100,180,255,0.08); border: 1px solid rgba(100,180,255,0.25);
    border-radius: 12px; padding: 0.8rem 1.2rem; color: #90c8ff;
    font-size: 0.85rem; margin-bottom: 1rem;
}

/* Footer */
.footer { text-align: center; color: #55556a; font-size: 0.78rem; padding: 2rem 0 1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — ML MODEL  (train once, cache for the session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    """
    Train a Linear Regression model on synthetic Indian housing data.
    Returns the trained model and evaluation metrics.
    No file is written to disk — everything lives in memory.
    """
    # ── Generate synthetic dataset ──────────────────────────────────
    rng = np.random.default_rng(42)
    n   = 1000

    area      = rng.integers(400, 5001, n)
    bedrooms  = rng.integers(1, 7, n)
    bathrooms = np.minimum(rng.integers(1, 6, n), bedrooms)   # bathrooms ≤ bedrooms

    price = (
        area      * 3_000          # ₹3,000 per sq.ft base rate
        + bedrooms  * 2_00_000     # bedroom premium
        + bathrooms * 1_50_000     # bathroom premium
        + rng.normal(0, 2_00_000, n)  # market noise
    )
    price = np.maximum(price, 5_00_000).astype(int)           # floor ₹5L

    X = np.column_stack([area, bedrooms, bathrooms])
    y = price

    # ── Train / test split ──────────────────────────────────────────
    idx     = rng.permutation(n)
    split   = int(n * 0.8)
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    # ── Train ───────────────────────────────────────────────────────
    model = LinearRegression()
    model.fit(X_train, y_train)

    # ── Evaluate ────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    return model, mae, r2


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — HELPER
# ─────────────────────────────────────────────────────────────────────────────
def format_inr(amount: float) -> str:
    """Format a rupee amount into Lakhs / Crores notation."""
    amount = int(round(amount, -3))          # round to nearest ₹1,000
    if amount >= 1_00_00_000:
        return f"₹{amount / 1_00_00_000:.2f} Cr"
    elif amount >= 1_00_000:
        return f"₹{amount / 1_00_00_000:.2f} L"
    return f"₹{amount:,}"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — UI
# ─────────────────────────────────────────────────────────────────────────────

# Load / train model
with st.spinner("Loading ML model…"):
    model, mae, r2 = get_model()

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏠 House Price Prediction</h1>
    <p>Enter your home's details below and our Machine Learning model will
       estimate the market price instantly — no sign-up required.</p>
</div>
""", unsafe_allow_html=True)

# ── Model info badge ─────────────────────────────────────────────────────────
st.markdown(
    f'<div class="info-box">✅ Model ready &nbsp;·&nbsp; '
    f'R² = <strong>{r2:.4f}</strong> &nbsp;·&nbsp; '
    f'MAE ≈ <strong>{format_inr(mae)}</strong></div>',
    unsafe_allow_html=True,
)

# ── Input card (Fixed via native st.container with custom css) ───────────────
st.markdown('<div class="section-label">Property Details</div>', unsafe_allow_html=True)

# Use Streamlit's native container wrapper to prevent DOM structure breakage
with st.container():
    # Injecting the card style class globally to target this native container layout
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input(
            "Area (sq.ft)",
            min_value=100, max_value=20_000, value=1200, step=50,
            help="Total built-up area of the house in square feet.",
        )

    with col2:
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1, max_value=10, value=3, step=1,
            help="Total number of bedrooms.",
        )

    with col3:
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1, max_value=10, value=2, step=1,
            help="Total number of bathrooms.",
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ── Predict button ───────────────────────────────────────────────────────────
predict_clicked = st.button("🔮  Predict House Price")

if predict_clicked:
    errors = []
    if area < 100:
        errors.append("Area must be at least 100 sq.ft.")
    if bathrooms > bedrooms:
        errors.append("Number of bathrooms cannot exceed number of bedrooms.")

    if errors:
        for e in errors:
            st.markdown(f'<div class="error-box">⚠️ {e}</div>', unsafe_allow_html=True)
    else:
        features  = np.array([[area, bedrooms, bathrooms]])
        price     = float(model.predict(features)[0])
        price     = max(price, 1_00_000)          # safety floor
        price_str = format_inr(price)

        st.markdown(f"""
        <div class="result-box">
            <div class="label">Estimated Market Price</div>
            <div class="price">{price_str}</div>
            <div class="summary">
                Area: <strong>{area:,} sq.ft</strong> &nbsp;|&nbsp;
                Bedrooms: <strong>{bedrooms}</strong> &nbsp;|&nbsp;
                Bathrooms: <strong>{bathrooms}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Built with Python · Streamlit · Scikit-learn &nbsp;|&nbsp; '
    'Predictions are estimates based on a trained ML model.</div>',
    unsafe_allow_html=True,
)
