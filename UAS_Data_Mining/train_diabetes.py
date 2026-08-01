import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ===========================
# Split Data
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===========================
# Normalisasi
# ===========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

# ===========================
# Model
# ===========================

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

for nama, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("="*40)
    print(nama)

    print("Accuracy :", accuracy_score(y_test,pred))
    print("Precision:", precision_score(y_test,pred))
    print("Recall   :", recall_score(y_test,pred))
    print("F1 Score :", f1_score(y_test,pred))

    filename = nama.lower().replace(" ","_") + ".pkl"

    joblib.dump(model,"models/"+filename)