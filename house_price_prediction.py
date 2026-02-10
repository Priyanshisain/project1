# house_price_prediction.py
# My first proper ML project - California house prices
# Just run: python house_price_prediction.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# If you get SSL error on Windows when downloading dataset, uncomment these two lines:
# import certifi
# import os
# os.environ['SSL_CERT_FILE'] = certifi.where()

print("Loading data...")
data = fetch_california_housing(as_frame=True)
df = data.frame

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head())
print("Missing values:\n", df.isnull().sum())

# Quick plots to understand data
plt.figure(figsize=(8,5))
sns.histplot(df['MedHouseVal'], kde=True, color='teal')
plt.title("House price distribution")
plt.xlabel("Median House Value (×100k $)")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation matrix")
plt.show()

# Income vs price - looks important
plt.figure(figsize=(8,5))
sns.scatterplot(x='MedInc', y='MedHouseVal', data=df, alpha=0.5)
plt.title("Income vs House Price")
plt.show()

# ─── Prepare data ───
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("Train shape:", X_train.shape)
print("Test shape: ", X_test.shape)

# ─── Models ───

# Simple linear model (baseline)
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
yhat_lr = lr.predict(X_test_sc)

rmse_linear = np.sqrt(mean_squared_error(y_test, yhat_lr))
r2_linear   = r2_score(y_test, yhat_lr)

print("\nLinear Regression:")
print(f"RMSE: {rmse_linear:.4f}")
print(f"R²:   {r2_linear:.4f}")

# Random Forest - much better usually
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_sc, y_train)
yhat_rf = rf.predict(X_test_sc)

rmse_rf = np.sqrt(mean_squared_error(y_test, yhat_rf))
r2_rf   = r2_score(y_test, yhat_rf)

print("\nRandom Forest:")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R²:   {r2_rf:.4f}")

# Quick extra experiment: log transform target (prices are right-skewed)
y_train_log = np.log1p(y_train)          # log(1 + y)
y_test_log  = np.log1p(y_test)

rf_log = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_log.fit(X_train_sc, y_train_log)

yhat_log = rf_log.predict(X_test_sc)
yhat_log_original = np.expm1(yhat_log)   # back to normal scale

rmse_log = np.sqrt(mean_squared_error(y_test, yhat_log_original))
print(f"\nRandom Forest with log-target RMSE: {rmse_log:.4f}  (sometimes better, sometimes similar)")

# ─── Plots of predictions ───
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(y_test, yhat_lr, alpha=0.4, color='blue')
plt.plot([0,6], [0,6], 'r--')
plt.title("Linear Regression\nActual vs Predicted")
plt.xlabel("Actual price")
plt.ylabel("Predicted")

plt.subplot(1,2,2)
plt.scatter(y_test, yhat_rf, alpha=0.4, color='green')
plt.plot([0,6], [0,6], 'r--')
plt.title("Random Forest\nActual vs Predicted")
plt.xlabel("Actual price")
plt.ylabel("Predicted")

plt.tight_layout()
plt.show()

# Feature importance from Random Forest
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(9,6))
imp.plot(kind='barh', color='coral')
plt.title("Which features matter most? (Random Forest)")
plt.show()

# My own summary
print("\n" + "-"*50)
print("What I learned from this project")
print("-"*50)
print(f"Linear model RMSE: {rmse_linear:.3f}  |  R²: {r2_linear:.3f}")
print(f"Random Forest RMSE: {rmse_rf:.3f}  |  R²: {r2_rf:.3f}")
print("→ Forest is clearly better here")
print("Most important feature: Median Income (no surprise really)")
print("Location (lat/long) also helps a bit")
print("Log transform on price gave similar or slightly better RMSE sometimes")
print("\nNext I want to try: cross-validation + maybe XGBoost")