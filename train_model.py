import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load original CSV (or your current dataset)
df = pd.read_csv("cleaned_titanic.csv")  # <- can be original CSV too

# 🔹 CLEANING: remove non-numeric columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Create FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df = df.drop(['SibSp','Parch'], axis=1)

# Encode categorical variables
df = pd.get_dummies(df, columns=['Sex','Embarked'], drop_first=True)

# Features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")