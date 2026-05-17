# 🎬 MOVIE RATING PREDICTION PROJECT

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# 1. LOAD DATASET
# -------------------------------
df = pd.read_csv("movies.csv", encoding="ISO-8859-1")

print("First 5 rows:")
print(df.head())

# -------------------------------
# 2. DATA CLEANING
# -------------------------------
print("\nMissing values:")
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()

# -------------------------------
# 3. ENCODING TEXT DATA
# -------------------------------
le_genre = LabelEncoder()
le_director = LabelEncoder()
le_actor = LabelEncoder()

df["Genre"] = le_genre.fit_transform(df["Genre"])
df["Director"] = le_director.fit_transform(df["Director"])
df["Actor 1"] = le_actor.fit_transform(df["Actor 1"])

# -------------------------------
# 4. FEATURES & TARGET
# -------------------------------
X = df[["Genre", "Director", "Actor 1"]]
y = df["Rating"]

# -------------------------------
# 5. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. MODEL TRAINING
# -------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 7. PREDICTION
# -------------------------------
y_pred = model.predict(X_test)

print("\nSample Predictions:")
print(y_pred[:5])

# -------------------------------
# 8. EVALUATION
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance:")
print("Mean Absolute Error:", round(mae, 3))
print("Root Mean Squared Error:", round(rmse, 3))
