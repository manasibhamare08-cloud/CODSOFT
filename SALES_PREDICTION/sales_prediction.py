# 📈 SALES PREDICTION USING SIMPLE LINEAR REGRESSION

# -------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 2. LOAD DATASET
# -------------------------------
df = pd.read_csv("advertising.csv")

print("First 5 rows:")
print(df.head())

# -------------------------------
# 3. CHECK DATA
# -------------------------------
print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# -------------------------------
# 4. VISUALIZE RELATIONSHIP
# -------------------------------
sns.scatterplot(x="TV", y="Sales", data=df)
plt.title("TV Advertising vs Sales")
plt.show()

# -------------------------------
# 5. FEATURES & TARGET
# -------------------------------
X = df[["TV"]]      # only TV (Simple Linear Regression)
y = df["Sales"]

# -------------------------------
# 6. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100
)

# -------------------------------
# 7. TRAIN MODEL
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 8. MODEL PARAMETERS
# -------------------------------
print("\nModel Equation:")
print("Intercept (c):", round(model.intercept_, 4))
print("Coefficient (m):", round(model.coef_[0], 4))

print(
    "\nSales =",
    round(model.intercept_, 4),
    "+",
    round(model.coef_[0], 4),
    "* TV"
)

# -------------------------------
# 9. PREDICTIONS
# -------------------------------
y_pred = model.predict(X_test)

print("\nSample Predictions:")
print(y_pred[:5])

# -------------------------------
# 10. EVALUATION
# -------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("RMSE:", round(rmse, 3))
print("R² Score:", round(r2, 3))

# -------------------------------
# 11. PREDICT NEW VALUE
# -------------------------------
sample_tv = [[150]]
prediction = model.predict(sample_tv)

print("\nPredicted Sales for TV budget = 150:")
print(round(prediction[0], 2))