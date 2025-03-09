import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv("Data/stress_detection_data.csv")

# Drop 'Wake_Up_Time' and 'Bed_Time' columns
df = df.drop(columns=["Wake_Up_Time", "Bed_Time"], errors='ignore')

# Initialize the label encoder
encoder = LabelEncoder()

# List of categorical columns to encode
categorical_columns = ["Gender", "Occupation", "Marital_Status", "Smoking_Habit", 
                       "Meditation_Practice", "Exercise_Type", "Stress_Detection"]

# Encode categorical columns and store mappings
mappings = {}
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])
    mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

# Print the mappings for reference
print("Category Mappings:")
for col, mapping in mappings.items():
    print(f"{col}: {mapping}\n")

# Check if the changes have been applied
print("\nUpdated DataFrame Preview:")
print(df.head())

# Features (X) and target variable (y)
X = df.drop(columns=["Stress_Detection"])  # Features
y = df["Stress_Detection"]  # Target

# Split into training (80%) and testing (20%) datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)  # Check dataset size

# Hyperparameter tuning for Decision Tree
param_grid_dt = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Decision Tree Classifier and perform GridSearchCV
dt_model = DecisionTreeClassifier(random_state=42)
grid_search_dt = GridSearchCV(dt_model, param_grid_dt, cv=5, n_jobs=-1, verbose=0)
grid_search_dt.fit(X_train, y_train)

# Best parameters for Decision Tree
print(f"Best Decision Tree Model: {grid_search_dt.best_params_}")

# Train the best Decision Tree model
best_dt_model = grid_search_dt.best_estimator_
y_pred_dt = best_dt_model.predict(X_test)

# Decision Tree evaluation
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Model Accuracy: {dt_accuracy:.2f}")
print(classification_report(y_test, y_pred_dt))

# Hyperparameter tuning for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest Classifier and perform GridSearchCV
rf_model = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, n_jobs=-1, verbose=0)
grid_search_rf.fit(X_train, y_train)

# Best parameters for Random Forest
print(f"Best Random Forest Model: {grid_search_rf.best_params_}")

# Train the best Random Forest model
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# Random Forest evaluation
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Model Accuracy: {rf_accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred_rf))

# Ensemble Model (Voting Classifier)
ensemble_model = VotingClassifier(estimators=[('dt', best_dt_model), ('rf', best_rf_model)], voting='hard')
ensemble_model.fit(X_train, y_train)

# Ensemble evaluation
ensemble_y_pred = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_y_pred)
print(f"Ensemble Model Accuracy: {ensemble_accuracy * 100:.2f}%")

# Cross-validation scores for Decision Tree and Random Forest
dt_cv_scores = cross_val_score(best_dt_model, X, y, cv=5, scoring='accuracy')
print(f"Decision Tree Cross-Validation Accuracy: {dt_cv_scores.mean():.2f}")

rf_cv_scores = cross_val_score(best_rf_model, X, y, cv=5, scoring='accuracy')
print(f"Random Forest Cross-Validation Accuracy: {rf_cv_scores.mean():.2f}")

# Save the best model (Random Forest or Decision Tree, based on performance)
model_to_save = best_rf_model if rf_accuracy > dt_accuracy else best_dt_model
joblib.dump(model_to_save, "stress_prediction_model.pkl")

# Load and use the model later
loaded_model = joblib.load("stress_prediction_model.pkl")
new_prediction = loaded_model.predict([[30, 1, 150, 2, 7.0, 4.0, 0, 2.0, 4.0, 0, 0, 8, 1.0, 5, 1, 0, 120, 180, 90]])
print("Predicted Stress Level:", new_prediction)