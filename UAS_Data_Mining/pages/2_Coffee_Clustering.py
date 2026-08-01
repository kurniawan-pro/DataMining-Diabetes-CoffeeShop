import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

st.title("☕ Analisis Klaster Lokasi Gerai Kopi dan Deteksi Zona Sepi")

st.write("""
Halaman ini menggunakan algoritma K-Means untuk mengelompokkan lokasi
gerai kopi berdasarkan koordinat geografis.
""")

# ===========================
# Load Data
# ===========================

df = pd.read_csv(BASE_DIR / "coffee_shop.csv")

kmeans = joblib.load(BASE_DIR / "models" / "kmeans.pkl")

scaler = joblib.load(BASE_DIR / "models" / "cluster_scaler.pkl")

# ===========================
# Membuat Cluster
# ===========================

X = df[["Latitude","Longitude"]]

X_scaled = scaler.transform(X)

df["Cluster"] = kmeans.predict(X_scaled)

# ===========================
# Menentukan Zona Sepi
# Cluster dengan jumlah data paling sedikit dianggap zona sepi
# ===========================

jumlah_cluster = df["Cluster"].value_counts()

zona_sepi = jumlah_cluster.idxmin()

df["Kategori"] = df["Cluster"].apply(
    lambda x: "Zona Sepi" if x == zona_sepi else "Zona Ramai"
)

st.subheader("📄 Data Gerai Kopi")

st.dataframe(df)

# ===========================
# Scatter Plot
# ===========================

st.subheader("📍 Visualisasi Cluster")

fig, ax = plt.subplots(figsize=(8,6))

warna = ["red","blue","green","orange","purple"]

for cluster in sorted(df["Cluster"].unique()):

    subset = df[df["Cluster"] == cluster]

    ax.scatter(
        subset["Longitude"],
        subset["Latitude"],
        label=f"Cluster {cluster}",
        s=60
    )

ax.set_xlabel("Longitude")

ax.set_ylabel("Latitude")

ax.legend()

st.pyplot(fig)

# ===========================
# Prediksi Lokasi Baru
# ===========================

st.subheader("Prediksi Lokasi Baru")

lat = st.number_input("Latitude", format="%.6f")

lon = st.number_input("Longitude", format="%.6f")

if st.button("Prediksi Lokasi"):

    lokasi = scaler.transform([[lat,lon]])

    hasil = kmeans.predict(lokasi)[0]

    st.success(f"Lokasi masuk Cluster {hasil}")

    if hasil==zona_sepi:
        st.error("Lokasi diprediksi termasuk ZONA SEPI")
    else:
        st.success("Lokasi diprediksi termasuk ZONA RAMAI")