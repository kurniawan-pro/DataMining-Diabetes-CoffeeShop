import streamlit as st

st.set_page_config(
    page_title="UAS Data Mining",
    page_icon="📊",
    layout="wide"
)

st.title("📊 UAS Data Mining")

st.markdown("---")

st.header("Prediksi Risiko Diabetes dan Analisis Klaster Gerai Kopi")

st.write("""
Aplikasi ini merupakan implementasi dua metode Data Mining dalam satu aplikasi Streamlit.

## Menu

Silakan pilih menu pada sidebar:

- 🩺 Prediksi Diabetes
- ☕ Clustering Gerai Kopi

""")

st.markdown("---")

st.subheader("Tentang Proyek")

st.write("""
### Bagian 1 - Klasifikasi Diabetes

Menggunakan tiga algoritma Machine Learning:

- K-Nearest Neighbor (KNN)
- Naive Bayes
- Decision Tree

Model digunakan untuk memprediksi apakah pasien berpotensi mengidap diabetes berdasarkan data kesehatan pasien.

---

### Bagian 2 - Clustering Gerai Kopi

Menggunakan algoritma K-Means untuk mengelompokkan lokasi gerai kopi berdasarkan:

- Koordinat lokasi (x dan y)
- Population Density
- Traffic Flow
- Competitor Count
- Commercial Area

Hasil clustering digunakan untuk membantu mengidentifikasi zona ramai maupun zona sepi.
""")

st.markdown("---")

st.success("Pilih menu pada sidebar untuk memulai.")