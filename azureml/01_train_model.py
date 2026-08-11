import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# Load the sample data
df = pd.read_csv("../data/avionics_sample.csv")

feature_columns = ['cycle', 'setting1', 'setting2', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7']
X = df[feature_columns]
y_cls = df['failure_imminent']
y_reg = df['RUL']  

X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlflow.set_experiment("avionics-predictive-maintenance")

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    clf.fit(X_train_scaled, y_cls_train)
    cls_report = classification_report(y_cls_test, clf.predict(X_test_scaled), output_dict=True)

    reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    reg.fit(X_train_scaled, y_reg_train)
    mae = mean_absolute_error(y_reg_test, reg.predict(X_test_scaled))
    r2 = r2_score(y_reg_test, reg.predict(X_test_scaled))

    mlflow.log_metrics({
        "classification_accuracy": cls_report['accuracy'],
        "rul_mae": mae,
        "rul_r2": r2
    })
    mlflow.sklearn.log_model(clf, "failure_classifier")
    mlflow.sklearn.log_model(reg, "rul_regressor")

    print(f"Accuracy: {cls_report['accuracy']:.3f}")
    print(f"RUL MAE: {mae:.1f} cycles")
    print(f"RUL R²: {r2:.3f}")