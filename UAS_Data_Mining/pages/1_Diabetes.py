import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

st.title("🩺 Prediksi Risiko Diabetes Berdasarkan Data Pasien")

st.write("""
Halaman ini digunakan untuk memprediksi apakah seorang pasien berisiko
mengidap diabetes menggunakan algoritma Machine Learning.
""")

# ==========================
# PILIH MODEL
# ==========================

st.subheader("Pilih Model")

model_name = st.selectbox(
    "Algoritma",
    (
        "KNN",
        "Naive Bayes",
        "Decision Tree"
    )
)

if model_name == "KNN":
    model = joblib.load(BASE_DIR / "models" / "knn.pkl")

elif model_name == "Naive Bayes":
    model = joblib.load(BASE_DIR / "models" / "naive_bayes.pkl")

else:
    model = joblib.load(BASE_DIR / "models" / "decision_tree.pkl")

scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

# ==========================
# INPUT DATA
# ==========================

st.subheader("Masukkan Data Pasien")

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input("Pregnancies", 0, 20, 1)

    glucose = st.number_input("Glucose", 0, 300, 120)

    blood = st.number_input("Blood Pressure", 0, 200, 70)

    skin = st.number_input("Skin Thickness", 0, 100, 20)

with col2:

    insulin = st.number_input("Insulin", 0, 900, 80)

    bmi = st.number_input("BMI", 0.0, 70.0, 25.5)

    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.5)

    age = st.number_input("Age", 1, 120, 30)

# ==========================
# PREDIKSI
# ==========================

if st.button("Prediksi"):

    data = np.array([[
        pregnancies,
        glucose,
        blood,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]])

    data = scaler.transform(data)

    hasil = model.predict(data)

    st.markdown("---")

    st.subheader("Hasil Prediksi")

    if hasil[0] == 1:

        st.error("⚠ Pasien diprediksi BERISIKO DIABETES")

    else:

        st.success("✅ Pasien diprediksi TIDAK BERISIKO DIABETES")

# ==========================
# DATASET
# ==========================

st.markdown("---")

st.subheader("📄 Dataset Diabetes")

try:
    df = pd.read_csv(BASE_DIR / "diabetes.csv")
    st.dataframe(df, use_container_width=True)

except FileNotFoundError:
    st.error("File diabetes.csv tidak ditemukan.")

# ==========================
# EVALUASI MODEL
# ==========================

st.markdown("---")

st.subheader("📊 Perbandingan Model")

try:

    evaluation = pd.read_csv(BASE_DIR / "evaluation" / "evaluation.csv")

    st.dataframe(evaluation, use_container_width=True)

except:

    st.warning("evaluation.csv belum tersedia.")

# ==========================
# CONFUSION MATRIX
# ==========================

st.markdown("---")

st.subheader("📌 Confusion Matrix")

if model_name == "KNN":

    cm_file = "knn_cm.csv"

elif model_name == "Naive Bayes":

    cm_file = "naive_bayes_cm.csv"

else:

    cm_file = "decision_tree_cm.csv"

try:

    cm = pd.read_csv(BASE_DIR / "evaluation" / cm_file)

    fig, ax = plt.subplots(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(model_name)

    st.pyplot(fig)

except:

    st.warning("Confusion Matrix belum tersedia.")