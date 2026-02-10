import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model + feature columns
# -----------------------------
model = joblib.load("model.pkl")
feature_cols = joblib.load("feature_cols.pkl")

st.set_page_config(page_title="Bank Deposit Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Bank Term Deposit Predictor")
st.write(
    "Answer a few quick questions and I’ll predict whether the customer is likely to subscribe "
    "to a term deposit (**deposit = yes/no**)."
)

# Debug info (helps you verify model is correct)
st.caption(f"✅ Model classes detected: {list(model.classes_)}")

st.divider()
st.subheader("📝 Customer Details (Quick Form)")

# -----------------------------
# USER-FRIENDLY INPUTS (short + important)
# -----------------------------
age = st.number_input("👤 What is the customer's age?", min_value=18, max_value=100, value=35)

balance = st.number_input(
    "💰 What is the customer's account balance?",
    value=0,
    help="Balance is one of the strongest predictors in the dataset."
)

contact = st.selectbox(
    "📞 How was the customer contacted?",
    ["cellular", "telephone", "unknown"]
)

day = st.slider(
    "📅 What day of the month was the customer contacted?",
    min_value=1, max_value=31, value=15
)

month = st.selectbox(
    "🗓️ What month was the customer contacted?",
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"]
)

campaign = st.number_input(
    "🔁 How many times was the customer contacted in this campaign?",
    min_value=1, value=1
)

housing = st.selectbox("🏠 Does the customer have a housing loan?", ["yes", "no"])
marital = st.selectbox("💍 What is the customer's marital status?", ["married", "single", "divorced"])

st.divider()

# -----------------------------
# HIDDEN DEFAULTS (NOT SHOWN TO USER)
# These keep the model input consistent with training.
# -----------------------------
DEFAULTS = {
    "job": "unknown",
    "education": "secondary",
    "default": "no",
    "loan": "no",
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}

# Build the raw input row (same raw columns as your training X before OHE)
df_input = pd.DataFrame([{
    "age": age,
    "job": DEFAULTS["job"],
    "marital": marital,
    "education": DEFAULTS["education"],
    "default": DEFAULTS["default"],
    "balance": balance,
    "housing": housing,
    "loan": DEFAULTS["loan"],
    "contact": contact,
    "day": day,
    "month": month,
    "campaign": campaign,
    "pdays": DEFAULTS["pdays"],
    "previous": DEFAULTS["previous"],
    "poutcome": DEFAULTS["poutcome"]
}])

# One-hot encode + align to training features
df_ohe = pd.get_dummies(df_input, drop_first=True)
df_ohe = df_ohe.reindex(columns=feature_cols, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("✅ Predict"):
    # Get probabilities
    proba = model.predict_proba(df_ohe)[0]
    p_no = proba[0]
    p_yes = proba[1]

    # Show probabilities (for transparency)
    st.write("Prediction probabilities:", {
        "no": float(p_no),
        "yes": float(p_yes)
    })

    # Custom threshold
    threshold = 0.55

    # Final decision
    if p_yes >= threshold:
        st.success("✅ Prediction: YES — The customer is likely to subscribe to a term deposit.")
    else:
        st.warning("❌ Prediction: NO — The customer is unlikely to subscribe to a term deposit.")

    st.caption(
        f"Decision threshold set at {threshold}. "
        "Predictions are based on probability rather than default class labels."
    )


# -----------------------------
# Sanity test (checks if model can output both classes)
# -----------------------------
if st.button("🧪 Run sanity test (2 cases)"):
    test_rows = pd.DataFrame([
        # Likely NO profile
        {"age": 25, "job": "unknown", "marital": "single", "education": "secondary",
         "default": "no", "balance": 0, "housing": "yes", "loan": "yes",
         "contact": "unknown", "day": 15, "month": "may", "campaign": 10,
         "pdays": -1, "previous": 0, "poutcome": "unknown"},

        # Likely YES profile
        {"age": 45, "job": "management", "marital": "married", "education": "tertiary",
         "default": "no", "balance": 5000, "housing": "no", "loan": "no",
         "contact": "cellular", "day": 5, "month": "may", "campaign": 1,
         "pdays": -1, "previous": 0, "poutcome": "unknown"}
    ])

    test_ohe = pd.get_dummies(test_rows, drop_first=True)
    test_ohe = test_ohe.reindex(columns=feature_cols, fill_value=0)

    preds = model.predict(test_ohe)
    st.write("✅ Sanity test predictions:", preds)

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(test_ohe)
        proba_maps = []
        for row_proba in probas:
            proba_maps.append({str(cls): float(p) for cls, p in zip(model.classes_, row_proba)})
        st.write("📊 Sanity test probabilities:", proba_maps)
