# train_model_wine.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# Convert wine quality (0–10) to binary classification
df['quality_label'] = df['quality'].apply(lambda q: 1 if q >= 7 else 0)

X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "wine_quality_model.pkl")
