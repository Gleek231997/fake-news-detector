import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Fake News Detector by Jaden", page_icon="🧠", layout="centered")

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.title("📝 How to Use")
st.sidebar.write("""
1. Enter a news headline or paragraph.
2. Click 🔍 Analyze to predict if it's Real or Fake.
3. Results appear instantly below!
""")
st.sidebar.markdown("---")
# Optional logo
# st.sidebar.image("logo.png", use_column_width=True)

# ------------------------------
# PAGE TITLE
# ------------------------------
st.title("🧠 Fake News Detector")
st.markdown("### Predict whether a news article is **Real** or **Fake** using a trained ML model.")
st.write("Model trained on the [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news).")

# ------------------------------
# LOAD MODEL AND VECTORIZER
# ------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except:
        st.warning("⚠️ Model files not found. Using a demo model with balanced classes...")
        # Demo dataset with BOTH classes
        demo_texts = [
            "Government announces new economic reforms.",     # Real
            "Stock markets hit record high this week.",       # Real
            "Scientists discover humans can photosynthesize.",# Fake
            "Aliens found living inside volcano on Mars."     # Fake
        ]
        demo_labels = [0, 0, 1, 1]  # 0 = Real, 1 = Fake

        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        X = vectorizer.fit_transform(demo_texts)

        model = PassiveAggressiveClassifier(max_iter=50)
        model.fit(X, demo_labels)

        return model, vectorizer

model, vectorizer = load_model()

# ------------------------------
# INPUT & RESULT COLUMNS
# ------------------------------
col1, col2 = st.columns([3, 2])

with col1:
    user_input = st.text_area("📰 Enter news text here:", placeholder="Type or paste a news article...")

with col2:
    st.markdown("### Prediction")
    if st.button("🔍 Analyze"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]
            if prediction == 0:
                st.markdown(f"<h2 style='color:green'>🟢 REAL NEWS</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color:red'>🔴 FAKE NEWS</h2>", unsafe_allow_html=True)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center'>🧩 Project by <a href='https://github.com/Gleek231997'>Glory Ekbote</a> | Built with Streamlit & Scikit-learn</p>",
    unsafe_allow_html=True
)
