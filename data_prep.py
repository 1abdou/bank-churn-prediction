from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "customer_churn.csv"
TARGET = "churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent 
OUTPUT_DIR ="processed_data"


# 1. Load the dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower()
df = df[df["age"] != 92] # Dropping an outlier

# 2. Separate features and target
X = df.drop(columns=TARGET).astype("float64")
y = df[TARGET]

# 3. Stratified train/test split (done BEFORE scaling to avoid data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
print("\nTarget distribution (train):")
print(y_train.value_counts(normalize=True).round(3))
print("\nTarget distribution (test):")
print(y_test.value_counts(normalize=True).round(3))

# 4. Scale continuous features only (skip binary / one-hot encoded columns)
continuous_cols = ['creditscore', 'age', 'tenure', 'balance', 'numofproducts', 'estimatedsalary']
print(f"\nScaling {len(continuous_cols)} continuous columns: {continuous_cols}")

scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])  # fit on train only
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])  # reuse train statistics

# 5. Save the processed splits and the fitted scaler
X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)
joblib.dump(scaler,f"{OUTPUT_DIR}/scaler.joblib")

print(f"\nSaved processed data and scaler to: {OUTPUT_DIR}")