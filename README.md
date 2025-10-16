# 🧠 Fake News Detector 

This project predicts whether a given news headline or article is **Real** or **Fake** using a trained machine learning model.  
The app is built with **Streamlit** for an interactive web interface.

---

## 🚀 Features
- 🔍 Detects fake vs real news in real-time  
- 🧠 Uses a pretrained **TF-IDF Vectorizer** + **Passive Aggressive Classifier**  
- 🌐 Built with **Streamlit** for easy web deployment  
- 🖌️ Modern UI with sidebar, columns, and colored prediction labels  

---

## 🧰 Tech Stack
- Python  
- Scikit-learn  
- Streamlit  
- Joblib  
- Pandas, NumPy  

---

## ⚙️ How to Run Locally

1. Clone the repository:


```bash
git clone git@github.com:Gleek231997/Fake-News-Detector-App.git
cd Fake-News-Detector-App

2. Install dependencies:
```bash
pip install streamlit scikit-learn pandas numpy joblib

3. Run the app:
```bash
streamlit run app.py

## 📁 Project Structure

```bash
Fake-News-Detector-App/
│
├── app.py              # Streamlit UI
├── model.pkl           # Pretrained classifier (optional)
├── vectorizer.pkl      # TF-IDF vectorizer (optional)
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies

