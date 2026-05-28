# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("attendance_data.csv")

# Drop ID if exists
if 'Student_ID' in df.columns:
    df.drop(columns=['Student_ID'], inplace=True)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split
X = df.drop("Dropout_Risk", axis=1)
y = df["Dropout_Risk"]

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save everything
joblib.dump(model, "model.pkl")
joblib.dump(label_encoders, "encoders.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("✅ Model training complete. Files saved.")