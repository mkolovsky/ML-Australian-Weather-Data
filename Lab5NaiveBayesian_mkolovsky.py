# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Weather Dataset
file_path = "weatherAUS.csv"
df = pd.read_csv(file_path)

# 2. Select Features and Target Variable
# Features: Selecting numerical columns relevant for classification
features = ['Rainfall', 'Humidity3pm', 'WindGustSpeed', 'Pressure3pm', 'Cloud3pm']
target = 'RainTomorrow'  # The classification target (predicting if it will rain)

# Drop rows with missing values to avoid issues in training
df = df[features + [target]].dropna()

# Convert categorical target ('Yes'/'No') to binary (1/0)
df[target] = df[target].map({'Yes': 1, 'No': 0})

# Display the first five rows of the dataset
print("First five rows of the weather dataset:")
print(df.head())

# 3. Split the dataset into training and testing sets (80% train, 20% test)
X = df[features].values  # Convert DataFrame to NumPy array for scikit-learn
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Initialize the Gaussian Naïve Bayes classifier
gnb = GaussianNB()

# Train the classifier
gnb.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = gnb.predict(X_test)

# 6. Evaluate the classifier's performance
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
cr = classification_report(y_test, y_pred, target_names=['No Rain', 'Rain'])
print(cr)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 7. Visualize the confusion matrix using a heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Rain', 'Rain'],
            yticklabels=['No Rain', 'Rain'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Gaussian Naïve Bayes')
plt.show()
