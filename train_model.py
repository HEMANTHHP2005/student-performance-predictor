import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Features and target
X = df[["study_hours", "attendance", "previous_score", "sleep_hours"]]
y = df["grade"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"MAE  : {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2   : {r2_score(y_test, y_pred):.2f}")

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")