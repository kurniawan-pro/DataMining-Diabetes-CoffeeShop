import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(page_title="Prediksi Diabetes", page_icon="🩺")

BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================
# Load Model
# ==========================
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

models = {
    "KNN": joblib.load(BASE_DIR / "models" / "knn.pkl"),
    "Naive Bayes": joblib.load(BASE_DIR / "models" / "naive_bayes.pkl"),
    "Decision Tree": joblib.load(BASE_DIR / "models" / "decision_tree.pkl")
}

# ==========================
# Judul
# ==========================
st.title("🩺 Prediksi Risiko Diabetes")

st.write("""
Halaman ini digunakan untuk memprediksi apakah seorang pasien
berpotensi mengidap diabetes menggunakan model Machine Learning.
""")

st.divider()

# ==========================
# Pilih Model
# ==========================
model_name = st.selectbox(
    "Pilih Model",
    list(models.keys())
)

model = models[model_name]

# ==========================
# Input Data
# ==========================

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        value=120
    )

    blood = st.number_input(
        "Blood Pressure",
        min_value=0,
        value=70
    )

    skin = st.number_input(
        "Skin Thickness",
        min_value=0,
        value=20
    )

with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        value=25.0
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        value=30
    )

# ==========================
# Prediksi
# ==========================

if st.button("Prediksi Diabetes"):

    data = [[
        pregnancies,
        glucose,
        blood,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]]

    data = scaler.transform(data)

    hasil = model.predict(data)[0]

    st.subheader("Hasil Prediksi")

    if hasil == 1:
        st.error("⚠️ Pasien diprediksi mengidap Diabetes")
    else:
        st.success("✅ Pasien diprediksi Tidak Diabetes")

st.divider()

# ==========================
# Evaluasi Model
# ==========================

st.subheader("Perbandingan Model")

hasil = pd.read_csv(BASE_DIR / "models" / "hasil_evaluasi.csv")

st.dataframe(hasil, use_container_width=True)

st.divider()

# ==========================
# Confusion Matrix
# ==========================

st.subheader("Confusion Matrix")

if model_name == "KNN":
    st.image(BASE_DIR / "models" / "KNN_cm.png")

elif model_name == "Naive Bayes":
    st.image(BASE_DIR / "models" / "Naive_Bayes_cm.png")

else:
    st.image(BASE_DIR / "models" / "Decision_Tree_cm.png")