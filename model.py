import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

# Load dataset
df = pd.read_csv("C:/Users/DELL/Desktop/Dermatology_App/dermatology.data", header=None)

# Replace '?' with NaN
df = df.replace('?', np.nan)

# Convert all columns to numeric
df = df.apply(pd.to_numeric)

# Handle missing values (fill with column mean)
df = df.fillna(df.mean())

# Prepare data
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as model.pkl successfully!")
