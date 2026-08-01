import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("coffee_shop.csv")

# Pilih fitur numerik, sesuaikan dengan nama kolom dataset
X = df[["Latitude", "Longitude"]]

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Simpan model dan scaler
joblib.dump(kmeans, "models/kmeans.pkl")
joblib.dump(scaler, "models/cluster_scaler.pkl")

# Tambahkan label cluster ke dataset
df["Cluster"] = kmeans.labels_

print(df.head())