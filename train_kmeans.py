import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("\n==============================")
print("TRAINING MODEL CLUSTERING")
print("==============================")

coffee = pd.read_csv("datasets/coffee_shop.csv")

print(coffee.columns)

print(coffee.head())
print(coffee.info())

print(coffee.columns)

features = [
    "x",
    "y",
    "population_density",
    "traffic_flow",
    "competitor_count",
    "is_commercial"
]

X = coffee[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

joblib.dump(
    scaler,
    "models/scaler_coffee.pkl"
)

inertia = []

for k in range(2, 8):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)

plt.figure(figsize=(6,4))

plt.plot(
    range(2,8),
    inertia,
    marker='o'
)

plt.xlabel("Jumlah Cluster")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.savefig("models/elbow_method.png")

plt.close()

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

joblib.dump(
    kmeans,
    "models/kmeans.pkl"
)

coffee["Cluster"] = kmeans.labels_

coffee.to_csv(
    "models/clustered_coffee.csv",
    index=False
)

score = silhouette_score(
    X_scaled,
    coffee["Cluster"]
)

print("\nSilhouette Score :", round(score,3))

plt.figure(figsize=(8,6))

plt.scatter(
    coffee["x"],
    coffee["y"],
    c=coffee["Cluster"],
    cmap="viridis"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Coffee Shop Clustering")

plt.savefig("models/coffee_cluster.png")

plt.close()

cluster_summary = coffee.groupby("Cluster").agg({
    "population_density": "mean",
    "traffic_flow": "mean",
    "competitor_count": "mean"
})

print(cluster_summary)