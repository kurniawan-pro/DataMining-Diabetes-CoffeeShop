import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Clustering Gerai Kopi",
    page_icon="☕"
)

BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================
# Load Model
# ==========================
kmeans = joblib.load(BASE_DIR / "models" / "kmeans.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler_coffee.pkl")

coffee = pd.read_csv(BASE_DIR / "models" / "clustered_coffee.csv")

# ==========================
# Judul
# ==========================
st.title("☕ Analisis Klaster Lokasi Gerai Kopi")

st.write("""
Halaman ini digunakan untuk menganalisis lokasi gerai kopi
menggunakan algoritma K-Means berdasarkan data spasial
dan parameter lingkungan.
""")

st.divider()

# ==========================
# Input
# ==========================

col1, col2 = st.columns(2)

with col1:

    x = st.number_input(
        "Koordinat X",
        value=50.0
    )

    y = st.number_input(
        "Koordinat Y",
        value=50.0
    )

    population = st.number_input(
        "Population Density",
        value=5000.0
    )

with col2:

    traffic = st.number_input(
        "Traffic Flow",
        value=1000.0
    )

    competitor = st.number_input(
        "Competitor Count",
        value=3
    )

    commercial = st.selectbox(
        "Commercial Area",
        [0,1]
    )

# ==========================
# Prediksi
# ==========================

if st.button("Analisis Lokasi"):

    data = [[
        x,
        y,
        population,
        traffic,
        competitor,
        commercial
    ]]

    data = scaler.transform(data)

    cluster = kmeans.predict(data)[0]

    st.subheader("Hasil Analisis")

    st.success(f"Lokasi termasuk Cluster {cluster}")

    # Menentukan zona
    summary = coffee.groupby("Cluster")["population_density"].mean()

    zona_sepi = summary.idxmin()

    if cluster == zona_sepi:
        st.warning("📍 Zona Sepi")
    else:
        st.success("📍 Zona Ramai")

st.divider()

# ==========================
# Informasi Cluster
# ==========================

st.subheader("Rata-rata Setiap Cluster")

summary = coffee.groupby("Cluster").agg({
    "population_density":"mean",
    "traffic_flow":"mean",
    "competitor_count":"mean"
})

st.dataframe(summary)

st.divider()

# ==========================
# Visualisasi Cluster
# ==========================

st.subheader("Visualisasi Clustering")

st.image(
    BASE_DIR / "models" / "coffee_cluster.png",
    use_container_width=True
)

st.divider()

st.subheader("Elbow Method")

st.image(
    BASE_DIR / "models" / "elbow_method.png",
    use_container_width=True
)