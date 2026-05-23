# 💳 CREDIT CARD FRAUD DETECTION

# -------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# 2. LOAD DATASET
# -------------------------------
df = pd.read_csv("creditcard.csv")

print("First 5 rows:")
print(df.head())

# -------------------------------
# 3. CHECK DATA
# -------------------------------
print("\nDataset Shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["Class"].value_counts())

# -------------------------------
# 4. VISUALIZE CLASS DISTRIBUTION
# -------------------------------
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Genuine Transactions")
plt.show()

# -------------------------------
# 5. HANDLE CLASS IMBALANCE
#    (Undersampling)
# -------------------------------
fraud = df[df["Class"] == 1]
normal = df[df["Class"] == 0]

normal_sample = normal.sample(n=len(fraud), random_state=42)

balanced_df = pd.concat([fraud, normal_sample])

print("\nBalanced Class Distribution:")
print(balanced_df["Class"].value_counts())

# -------------------------------
# 6. FEATURES & TARGET
# -------------------------------
X = balanced_df.drop("Class", axis=1)
y = balanced_df["Class"]

# -------------------------------
# 7. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 8. TRAIN MODEL
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# 9. PREDICTIONS
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 10. EVALUATION
# -------------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))