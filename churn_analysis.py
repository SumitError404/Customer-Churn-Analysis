import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import joblib

# Step 1: Load the dataset from CSV
df = pd.read_csv("churn_data.csv")

# Step 2: SQL filter to keep customers with tenure > 0
conn = sqlite3.connect(":memory:")
df.to_sql("customers", conn, index=False, if_exists="replace")
query = "SELECT * FROM customers WHERE tenure > 0"
df = pd.read_sql(query, conn)

# Debug print: check data size and class distribution
print(f"Dataset shape after filtering: {df.shape}")
print("Target value counts:\n", df['Churn'].value_counts())

# Step 3: Data Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.drop_duplicates(inplace=True)

# Step 4: Feature Engineering
df['AvgChargesPerMonth'] = df['TotalCharges'] / df['tenure']
df['AvgChargesPerMonth'].fillna(0, inplace=True)

# Step 5: Encode categorical variables (skip customerID)
le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    if col != 'customerID':
        df[col] = le.fit_transform(df[col])

# Step 6: Define features and target
X = df.drop(columns=['Churn', 'customerID'])
y = df['Churn']

# Step 7: Check dataset size and classes before splitting
if df.shape[0] < 10 or len(df['Churn'].unique()) < 2:
    print("Warning: Dataset too small or missing classes for stratified split.")
    stratify_param = None
else:
    stratify_param = y

# Perform train-test split, stratify only if possible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_param)

# Step 8: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 9: Define models
log_reg = LogisticRegression(random_state=42, max_iter=1000)
rf_clf = RandomForestClassifier(random_state=42)

# Step 10: Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"Best Random Forest parameters found: {grid_search.best_params_}")

# Step 11: Cross-validation for Logistic Regression
cv_scores = cross_val_score(log_reg, X_train, y_train, cv=5, scoring='accuracy')
print(f"Logistic Regression CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Step 12: Train models
log_reg.fit(X_train, y_train)
best_rf.fit(X_train, y_train)

# Step 13: Predict test
y_pred_lr = log_reg.predict(X_test)
y_pred_rf = best_rf.predict(X_test)

# Step 14: Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

evaluate_model(y_test, y_pred_lr, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# Step 15: Feature importance for Random Forest
feat_importances = pd.Series(best_rf.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_importances[:10], y=feat_importances.index[:10])
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

# Step 16: ROC curves
plt.figure(figsize=(8,6))
for model, y_pred_proba, label in zip(
    [log_reg, best_rf],
    [log_reg.predict_proba(X_test)[:,1], best_rf.predict_proba(X_test)[:,1]],
    ['Logistic Regression', 'Random Forest']
):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.3f})')

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Step 17: Save the model and scaler
joblib.dump(best_rf, "best_churn_model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("Model and scaler saved for deployment.")
