# 🌸 IRIS FLOWER CLASSIFICATION PROJECT

# -------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 2. LOAD DATASET
# -------------------------------
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target column (species)
df["species"] = iris.target

# Replace numbers with flower names
df["species"] = df["species"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

print("First 5 rows:")
print(df.head())

# -------------------------------
# 3. CHECK DATA
# -------------------------------
print("\nMissing values:")
print(df.isnull().sum())

# -------------------------------
# 4. VISUALIZATION
# -------------------------------
sns.pairplot(df, hue="species")
plt.show()

# -------------------------------
# 5. FEATURES & TARGET
# -------------------------------
X = df.drop("species", axis=1)
y = df["species"]

# -------------------------------
# 6. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 7. TRAIN MODEL
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 8. PREDICTION
# -------------------------------
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:10])

# -------------------------------
# 9. EVALUATION
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))