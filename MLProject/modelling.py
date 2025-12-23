import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# Load dataset
data = pd.read_csv("diabetes_preprocessing.csv")

X = data.drop(columns=["Outcome"])
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

# Log model to MLflow (ADVANCED)
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model"
)

print("Training completed successfully")
print("Accuracy:", acc)

