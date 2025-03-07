# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# -----------------------------
# SVM Classification for Weather Data (Optimized)
# -----------------------------

# 1. Load the Weather Dataset
file_path = "weatherAUS.csv"  # Your dataset
df = pd.read_csv(file_path)

# 2. Select Features and Target Variable
features = ['Rainfall', 'Humidity3pm', 'WindGustSpeed', 'Pressure3pm', 'Cloud3pm']
target = 'RainTomorrow'  # The classification target (predicting if it will rain)

# Drop rows with missing values to avoid errors in training
df = df[features + [target]].dropna()

# Convert categorical target ('Yes'/'No') to binary (1/0)
df[target] = df[target].map({'Yes': 1, 'No': 0})

# 3. Sample 5000 Rows for Faster Training
df_sampled = df.sample(n=5000, random_state=42)  # Reduce dataset size

# Display dataset summary
print(f"Dataset reduced to {df_sampled.shape[0]} samples for faster training.")
print("First five rows of the sampled dataset:")
print(df_sampled.head())

# 4. Split the dataset into training and testing sets (80% train, 20% test)
X = df_sampled[features].values  # Convert DataFrame to NumPy array for scikit-learn
y = df_sampled[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train SVM Model (Fixed Parameters for Faster Training)
svc = SVC(kernel='linear', C=1, gamma='scale')  # Using only one model (no GridSearch)
svc.fit(X_train, y_train)  # Train SVM model

# 6. Predict on Test Set
y_pred = svc.predict(X_test)

# 7. Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"SVC Classification Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Rain', 'Rain']))

# 8. Plot the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for SVM Classification (5000 Sample Model)')
plt.show()
