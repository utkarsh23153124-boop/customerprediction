# customerprediction
Customer Behaviour prediction and analysis of purchasing

# ======================================
# CREDIT RISK PREDICTION SYSTEM
# Logistic Regression + Random Forest
# ======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
import joblib


# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("Loan_Default.csv")

print("Dataset shape:", df.shape)


# ===============================
# 2. PREPROCESSING
# ===============================
y = df["Status"]
X = df.drop("Status", axis=1)

# Missing Values Handling

num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include='object').columns

# numeric → median
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

# categorical → mode
for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])


# one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# scaling
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])


# ===============================
# 3. TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# ===============================
# 4. HANDLE IMBALANCE (SMOTE)
# ===============================
print("Before SMOTE:\n", y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("After SMOTE:\n", y_train_sm.value_counts())


# ===============================
# 5A. LOGISTIC REGRESSION (GridSearch)
# ===============================
log_params = {
    'C': [0.01, 0.1, 1, 10],
    'class_weight': ['balanced']
}

log_grid = GridSearchCV(
    LogisticRegression(max_iter=3000),
    log_params,
    cv=3,
    scoring='f1'
)

log_grid.fit(X_train_sm, y_train_sm)
best_log = log_grid.best_estimator_

print("Best Logistic Params:", log_grid.best_params_)


# ===============================
# 5B. RANDOM FOREST (GridSearch)
# ===============================
rf_params = {
    'n_estimators': [150, 250],
    'max_depth': [10, 20],
    'class_weight': ['balanced']
}

rf_grid = GridSearchCV(
    RandomForestClassifier(),
    rf_params,
    cv=3,
    scoring='f1',
    n_jobs=1
)

rf_grid.fit(X_train_sm, y_train_sm)
best_rf = rf_grid.best_estimator_

print("Best RF Params:", rf_grid.best_params_)


# ===============================
# 6. EVALUATION FUNCTION
# ===============================
def evaluate(model, name):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    print("\n========", name, "========")
    print("Accuracy :", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall   :", recall_score(y_test, pred))
    print("F1 Score :", f1_score(y_test, pred))
    print("ROC-AUC  :", roc_auc_score(y_test, prob))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))


# ===============================
# 7. COMPARE MODELS
# ===============================
evaluate(best_log, "Logistic Regression")
evaluate(best_rf, "Random Forest")


# ===============================
# 8. FEATURE IMPORTANCE (RF)
# ===============================
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()


# ===============================
# 9. SELECT BEST MODEL
# ===============================
log_f1 = f1_score(y_test, best_log.predict(X_test))
rf_f1 = f1_score(y_test, best_rf.predict(X_test))

if rf_f1 > log_f1:
    final_model = best_rf
    print("\n✅ Final Model: Random Forest")
else:
    final_model = best_log
    print("\n✅ Final Model: Logistic Regression")

