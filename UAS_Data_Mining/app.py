import streamlit as st

# Konfigurasi halaman
st.set_page_config(
    page_title="Data Mining Project",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("📚 Menu")
st.sidebar.info(
    """
    **Data Mining Project**

    Pilih menu pada sidebar:

    🏠 Home

    🩺 Prediksi Diabetes

    ☕ Clustering Gerai Kopi
    """
)

# Judul
st.title("📊 Data Mining Project")

st.markdown("---")

# Informasi Mahasiswa
col1, col2 = st.columns(2)

with col1:
    st.subheader("👨‍🎓 Mahasiswa")
    st.write("**Nama :** Kurniawan Syahputra")
    st.write("**NIM :** 23146019")
    st.write("**Mata Kuliah :** Data Mining")

with col2:
    st.subheader("🎯 Tujuan Proyek")
    st.write("""
    Proyek ini bertujuan mengimplementasikan dua teknik Data Mining, yaitu:

    - Klasifikasi menggunakan:
        - K-Nearest Neighbor (KNN)
        - Naïve Bayes
        - Decision Tree

    - Clustering menggunakan:
        - K-Means
    """)

st.markdown("---")

st.header("📌 Deskripsi Proyek")

st.write("""
Aplikasi ini terdiri dari dua modul utama:

### 🩺 1. Prediksi Risiko Diabetes
Model Machine Learning digunakan untuk memprediksi apakah seorang pasien berisiko
mengidap diabetes berdasarkan data medis seperti Glucose, BMI, Age,
Blood Pressure, dan lainnya.

Model yang dibandingkan:

- KNN
- Naïve Bayes
- Decision Tree

Output yang ditampilkan:

- Prediksi Diabetes
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

### ☕ 2. Analisis Klaster Gerai Kopi

Menggunakan algoritma K-Means untuk mengelompokkan lokasi gerai kopi berdasarkan
koordinat geografis sehingga dapat diketahui zona yang ramai maupun zona sepi.

Output:

- Scatter Plot Cluster
- Prediksi Lokasi Baru
- Zona Ramai
- Zona Sepi
""")

st.markdown("---")

st.header("🛠 Algoritma yang Digunakan")

alg1, alg2, alg3, alg4 = st.columns(4)

alg1.metric("KNN", "Classification")
alg2.metric("Naïve Bayes", "Classification")
alg3.metric("Decision Tree", "Classification")
alg4.metric("K-Means", "Clustering")

st.markdown("---")

st.success("Silakan pilih menu pada sidebar untuk memulai.")