import streamlit as st

st.set_page_config(
    page_title="UAS Data Mining",
    page_icon="📊",
    layout="wide"
)

# ==========================
# Sidebar
# ==========================
with st.sidebar:
    st.title("📊 UAS Data Mining")

    st.markdown("### 👨‍🎓 Mahasiswa")

    st.write("**Nama:**")
    st.write("Kurniawan Syahputra")

    st.write("**NIM:**")
    st.write("23146019")

    st.write("**Mata Kuliah:**")
    st.write("Data Mining")

    st.divider()

    st.info("Silakan pilih menu pada sidebar Streamlit untuk membuka halaman Diabetes atau Coffee Clustering.")

# ==========================
# Halaman Utama
# ==========================
st.title("📊 UAS Data Mining")

st.markdown("---")

st.header("Prediksi Risiko Diabetes dan Analisis Klaster Gerai Kopi")

st.write("""
Aplikasi ini merupakan implementasi dua metode Data Mining:

1. 🩺 Prediksi Risiko Diabetes menggunakan:
   - KNN
   - Naive Bayes
   - Decision Tree

2. ☕ Analisis Klaster Gerai Kopi menggunakan:
   - K-Means Clustering
""")

st.success("Pilih menu pada sidebar untuk memulai.")