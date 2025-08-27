import joblib
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="SMS Spam Detector", page_icon="📱")

st.title("📱 SMS Spam Detector (Classic ML)")
st.write("Enter an SMS text and the model will predict whether it's **spam** or **ham** (not spam).")

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model" / "sms_spam_pipeline.joblib"
    return joblib.load(model_path)

model = load_model()

txt = st.text_area("Type or paste an SMS message:", height=150, placeholder="e.g., 'Congratulations! You won a prize. Click here...'")
if st.button("Predict"):
    if not txt.strip():
        st.warning("Please enter a message.")
    else:
        pred = model.predict([txt])[0]
        proba = max(model.predict_proba([txt])[0])
        st.subheader(f"Prediction: {'🚨 SPAM' if pred=='spam' else '✅ HAM'}")
        st.write(f"Confidence: {proba*100:.1f}%")
        st.caption("Model: TF-IDF + Logistic Regression (trained on synthetic SMS data)")
st.markdown("---")
st.caption("Built with scikit-learn + Streamlit")