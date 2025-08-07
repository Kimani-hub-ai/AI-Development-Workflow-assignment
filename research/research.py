import pandas as pd
import numpy as np

# Simulate patient dataset
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'age': np.random.randint(20, 90, n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'num_lab_procedures': np.random.randint(1, 100, n_samples),
    'num_medications': np.random.randint(1, 30, n_samples),
    'time_in_hospital': np.random.randint(1, 14, n_samples),
    'has_chronic_conditions': np.random.choice([0, 1], n_samples),
    'readmitted': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # target variable
})

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Handle categorical variables
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])  # Male=1, Female=0

# Normalize features
scaler = MinMaxScaler()
features = ['age', 'num_lab_procedures', 'num_medications', 'time_in_hospital']
data[features] = scaler.fit_transform(data[features])

# Separate features and labels
X = data.drop('readmitted', axis=1)
y = data['readmitted']

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split dataset: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Predict on validation set
y_pred = model.predict(X_val)

# Evaluate
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:\n", cm)

# Simulate concept drift by changing the distribution
drift_data = X_val.copy()
drift_data['has_chronic_conditions'] = 1  # All have chronic conditions

# Evaluate model on new drifted data
y_drift_pred = model.predict(drift_data)
drift_accuracy = accuracy_score(y_val, y_drift_pred)
print("Post-drift accuracy:", drift_accuracy)

import joblib

# Save the model
joblib.dump(model, 'readmission_predictor.pkl')

# Later... load the model
loaded_model = joblib.load('readmission_predictor.pkl')

from sklearn.model_selection import cross_val_score

# Use 5-fold cross-validation to evaluate model robustness
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-validated accuracy scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))
