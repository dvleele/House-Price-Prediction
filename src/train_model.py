import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
# Models
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------
#               Load data
df = pd.read_csv('../data/train.csv')
print(df.head())

# Select features
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath"]
X = df[features].fillna(0)
y = df["SalePrice"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------
#                Models
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f" Random Forest -> RMSE: {rmse:.2f}, R²: {r2:.2f}")

# ---------------------------------------
#          Save model
joblib.dump(rf, "house_price_model.pkl")
print(" Model đã được lưu thành công vào house_price_model.pkl")