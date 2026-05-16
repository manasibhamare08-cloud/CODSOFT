import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Show first 5 rows (optional)
print(data.head())

# Keep only useful columns
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# Fill missing age values with median
data['Age'] = data['Age'].fillna(data['Age'].median())

# Convert Sex column: male=0, female=1
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Features (X) and target (y)
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, predictions)

# Print result
print("\nTitanic Survival Prediction Model")
print("Model Accuracy:", round(accuracy * 100, 2), "%")