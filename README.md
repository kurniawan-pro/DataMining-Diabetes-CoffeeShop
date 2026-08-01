# UAS Data Mining
## Prediksi Risiko Diabetes dan Analisis Klaster Lokasi Gerai Kopi

### Nama
Kurniawan Syahputra

### NIM
23146019

---

## Deskripsi Proyek

Proyek ini merupakan implementasi dua metode Data Mining dalam satu aplikasi berbasis **Streamlit**, yaitu:

### 1. Prediksi Risiko Diabetes (Classification)

Model klasifikasi digunakan untuk memprediksi apakah seorang pasien berpotensi menderita diabetes berdasarkan data kesehatan pasien.

Algoritma yang digunakan:

- K-Nearest Neighbor (KNN)
- Naive Bayes
- Decision Tree

Parameter evaluasi:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Dataset:
Pima Indians Diabetes Database

---

### 2. Analisis Klaster Lokasi Gerai Kopi (K-Means)

Model clustering digunakan untuk mengelompokkan lokasi gerai kopi berdasarkan data spasial dan parameter lingkungan.

Fitur yang digunakan:

- x
- y
- population_density
- traffic_flow
- competitor_count
- is_commercial

Hasil clustering digunakan untuk membantu mengidentifikasi zona yang berpotensi ramai maupun zona sepi.

Evaluasi menggunakan:

- Elbow Method
- Silhouette Score

---

## Struktur Project

```
UAS_Data_Mining/
│
├── app.py
├── train_diabetes.py
├── train_kmeans.py
├── requirements.txt
├── README.md
│
├── datasets/
│   ├── diabetes.csv
│   └── coffee_shop.csv
│
├── models/
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── decision_tree.pkl
│   ├── scaler.pkl
│   ├── kmeans.pkl
│   ├── scaler_coffee.pkl
│   ├── hasil_evaluasi.csv
│   ├── coffee_cluster.png
│   └── elbow_method.png
│
├── pages/
│   ├── 1_Diabetes.py
│   └── 2_Coffee_Clustering.py
│
└── images/
```

---

## Library yang Digunakan

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Joblib

---

## Cara Menjalankan Project

### 1. Clone Repository

```bash
git clone https://github.com/username/UAS_Data_Mining.git
```

### 2. Masuk ke Folder Project

```bash
cd UAS_Data_Mining
```

### 3. Install Library

```bash
pip install -r requirements.txt
```

### 4. Training Model

Training klasifikasi diabetes:

```bash
python train_diabetes.py
```

Training clustering gerai kopi:

```bash
python train_kmeans.py
```

### 5. Jalankan Streamlit

```bash
streamlit run app.py
```

---

## Dataset

### Diabetes

- Pima Indians Diabetes Dataset

### Coffee Shop

- Dataset Gerai Kopi (UAS Data Mining)

---

## Deployment

### GitHub Repository

https://github.com/kurniawan-pro/DataMining-Diabetes-CoffeeShop.git

### Streamlit Cloud

https://datamining-diabetes-coffeeshop-brwkv5vawhaxsuhpv9ozth.streamlit.app/

---

## Hasil Project

Aplikasi terdiri dari dua halaman utama:

### Prediksi Diabetes

- Memilih model klasifikasi
- Input data pasien
- Prediksi diabetes
- Menampilkan metrik evaluasi
- Menampilkan confusion matrix

### Clustering Gerai Kopi

- Visualisasi hasil clustering
- Analisis lokasi baru
- Prediksi cluster
- Identifikasi zona ramai dan zona sepi

---

## Penulis

Nama : [Kurniawan Syahputra]

NIM : [23146019]

Mata Kuliah : Data Mining

Universitas : [Abulyatama]
