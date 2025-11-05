"""
Titanic Survival Prediction using Logistic Regression

This script predicts passenger survival on the Titanic using logistic regression.
It includes data preprocessing, visualization, and model evaluation.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Visualize survival count by gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Count by Gender")
plt.show()

# Handle missing values
df.drop('Cabin', axis=1, inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical features
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')

# Drop unnecessary columns
X = df.drop(['Survived', 'Name', 'Ticket'], axis=1)
Y = df['Survived']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Fit model
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(Y_test, Y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

# Detailed performance report
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

