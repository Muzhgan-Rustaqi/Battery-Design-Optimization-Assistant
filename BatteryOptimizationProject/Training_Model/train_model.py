# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# 1️⃣ Load dataset and cleat it.

df = pd.read_csv(r'C:\Users\muzhg\BatteryOptimizationProject\data.csv',encoding="latin1") 

df=df.rename(columns={"Unnamed: 8": "Battery Consumed (%)"})

battery_cols = [
    "Battery State of Charge (Start)",
    "Battery State of Charge (End)",
    "Battery Consumed (%)"
]

for col in battery_cols:
    # 1. Convert to string
    df[col] = df[col].astype(str)
    
    # 2. Remove % if present
    df[col] = df[col].str.replace("%", "", regex=False).str.strip()
    
    # 3. Convert to numeric, set errors to NaN
    df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 4. Fill missing values with mean
    df[col] = df[col].fillna(df[col].mean())

# 2️⃣ Features & target
features = [
    "Battery Temperature (Start) [°C]",
    "Battery Temperature (End)",
    "Battery State of Charge (Start)",
    "Battery State of Charge (End)",
    "Ambient Temperature (Start) [°C]",
    "Distance [km]",
    "Duration [min]",
    "Target Cabin Temperature",
]

target = "Battery Consumed (%)"

X = df[features].fillna(0)
y = df[target].fillna(df[target].mean())

# 3️⃣ Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Train model
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Evaluate
pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

# 6️⃣ Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_degradation.joblib")
print("Model saved successfully!")

#MAE & R²

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae, "R²:", r2)

#Calibration (predicted vs actual)
plt.scatter(y_test, pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual degradation")
plt.ylabel("Predicted degradation")
plt.title("Calibration plot")
plt.show()
