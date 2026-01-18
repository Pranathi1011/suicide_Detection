import streamlit as st
import pickle
import pandas as pd

# ----------------- Load All Models and Vectorizer (LOCAL FILES) -----------------
@st.cache_resource
def load_models_and_vectorizer():
    models = {}

    with open("rf_model.pkl", "rb") as f:
        models["Random Forest"] = pickle.load(f)

    with open("xgb_model.pkl", "rb") as f:
        models["XGBoost"] = pickle.load(f)

    with open("gb_model.pkl", "rb") as f:
        models["Gradient Boosting"] = pickle.load(f)

    with open("nb_model.pkl", "rb") as f:
        models["Naive Bayes"] = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return models, vectorizer


models, vectorizer = load_models_and_vectorizer()

# ----------------- Streamlit App -----------------
st.set_page_config(page_title="Suicide Risk Detection", layout="centered")

st.title("🧠 Suicide Risk Detection System")
st.subheader("AI-based Mental Health Text Analysis")

user_text = st.text_area(
    "Enter text to analyze:",
    height=150,
    placeholder="Type a message here..."
)

# ----------------- Prediction Function -----------------
def predict_text_all_models(text):
    X = vectorizer.transform([text])
    results = {}

    for name, model in models.items():
        pred = model.predict(X)[0]
        conf = max(model.predict_proba(X)[0])
        label = "Suicide" if pred == 1 else "Non-Suicide"

        results[name] = {
            "label": label,
            "confidence": conf
        }

    return results


# ----------------- Analyze Button -----------------
if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        results = predict_text_all_models(user_text)

        st.subheader("🔍 Model Predictions")

        table_data = []

        for model_name, res in results.items():
            label = res["label"]
            conf = res["confidence"]

            if label == "Suicide":
                st.error(f"{model_name}: ⚠️ Prediction: {label}")
            else:
                st.success(f"{model_name}: ✅ Prediction: {label}")

            st.info(f"{model_name} Confidence: {conf * 100:.2f}%")

            table_data.append({
                "Model": model_name,
                "Prediction": label,
                "Confidence (%)": round(conf * 100, 2)
            })

        # ----------------- Comparison Table -----------------
        df = pd.DataFrame(table_data)
        st.subheader("📊 Model-wise Prediction Comparison")
        st.dataframe(df, use_container_width=True)

# ----------------- Disclaimer -----------------
st.warning(
    "⚠️ This tool is for academic and research purposes only. "
    "It is not a medical diagnosis system."
)
