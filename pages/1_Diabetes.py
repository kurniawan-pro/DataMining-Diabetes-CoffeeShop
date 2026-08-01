import streamlit as st
import joblib
import numpy as np

st.title("Prediksi Risiko Diabetes")

model_name = st.selectbox(
    "Pilih Model",
    [
        "KNN",
        "Naive Bayes",
        "Decision Tree"
    ]
)

if model_name == "KNN":
    model = joblib.load("models/knn.pkl")

elif model_name == "Naive Bayes":
    model = joblib.load("models/naive_bayes.pkl")

else:
    model = joblib.load("models/decision_tree.pkl")

scaler = joblib.load("models/scaler.pkl")

preg = st.number_input("Pregnancies",0)

glucose = st.number_input("Glucose",0)

bp = st.number_input("Blood Pressure",0)

skin = st.number_input("Skin Thickness",0)

insulin = st.number_input("Insulin",0)

bmi = st.number_input("BMI",0.0)

dpf = st.number_input("Diabetes Pedigree Function",0.0)

age = st.number_input("Age",1)

if st.button("Prediksi"):

    data = np.array([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])

    data = scaler.transform(data)

    hasil = model.predict(data)

    if hasil[0]==1:
        st.error("Pasien diprediksi Mengidap Diabetes")
    else:
        st.success("Pasien diprediksi Tidak Diabetes")