import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# =========================
# Load dataset
# =========================
data = pd.read_csv("diabetes_preprocessing.csv")

X = data.drop(columns=["Outcome"])
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# MLflow config
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CI_RandomForest_Diabetes")

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)

    dump(model, "rf_model.joblib")
    mlflow.log_artifact("rf_model.joblib")

    print("Training completed. Accuracy:", acc)
