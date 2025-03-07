# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Weather Dataset
file_path = "weatherAUS.csv"
df = pd.read_csv(file_path)

# 2. Select Features and Target Variable
# Target: 'RainTomorrow' (Predicting if it will rain the next day)
df = df[['Rainfall', 'Humidity3pm', 'WindGustSpeed', 'Pressure3pm', 'Cloud3pm', 'RainTomorrow']].dropna()

# Convert 'RainTomorrow' (Yes/No) into binary (1 = Yes, 0 = No)
df['RainTomorrow'] = df['RainTomorrow'].replace({'Yes': 1, 'No': 0})

# Feature matrix (X) and target variable (y)
X = df.drop(columns=['RainTomorrow'])
y = df['RainTomorrow']

# Display the first five rows of the dataset
print("First five rows of the Weather dataset:")
print(df.head())

# 3. Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Initialize the Decision Tree classifier
dtc = DecisionTreeClassifier(
    criterion='gini',  # Splitting criterion
    max_depth=5,       # Maximum depth of the tree
    random_state=42
)

# Train the classifier
dtc.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = dtc.predict(X_test)

# 6. Evaluate the classifier's performance
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
cr = classification_report(y_test, y_pred, target_names=['No Rain', 'Rain'])
print(cr)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 7. Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(
    dtc,
    feature_names=X.columns,
    class_names=['No Rain', 'Rain'],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title('Decision Tree - Rain Prediction')
plt.show()

# 8. Visualize the confusion matrix using a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Rain', 'Rain'],
            yticklabels=['No Rain', 'Rain'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Decision Tree')
plt.show()
