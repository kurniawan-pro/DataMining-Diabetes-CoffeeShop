import pandas as pd
import numpy as np

import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay


df = pd.read_csv("datasets/diabetes.csv")
print(df.head())
print(df.shape)

print(df.isnull().sum())
print((df == 0).sum())

cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in cols:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop("Outcome", axis=1)

y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

nb = GaussianNB()

nb.fit(X_train, y_train)

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

joblib.dump(knn, "models/knn.pkl")

joblib.dump(nb, "models/naive_bayes.pkl")

joblib.dump(dt, "models/decision_tree.pkl")

def evaluate(model, name):
    y_pred = model.predict(X_test)

    print(f"\n===== {name} =====")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

evaluate(knn, "KNN")
evaluate(nb, "Naive Bayes")
evaluate(dt, "Decision Tree")

def show_confusion_matrix(model, name):
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        cmap="Blues"
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"models/{name}_cm.png")
    plt.close()

show_confusion_matrix(knn, "KNN")
show_confusion_matrix(nb, "Naive_Bayes")
show_confusion_matrix(dt, "Decision_Tree")

results = []

def evaluate(model, name):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 3),
        "Precision": round(pre, 3),
        "Recall": round(rec, 3),
        "F1-Score": round(f1, 3)
    })

    print(f"\n{name}")
    print("Accuracy :", acc)
    print("Precision:", pre)
    print("Recall   :", rec)
    print("F1 Score :", f1)

evaluate(knn, "KNN")
evaluate(nb, "Naive Bayes")
evaluate(dt, "Decision Tree")

result_df = pd.DataFrame(results)

print(result_df)

result_df.to_csv("models/hasil_evaluasi.csv", index=False)