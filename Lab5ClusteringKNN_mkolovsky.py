# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 1. Load the Weather Dataset
file_path = "weatherAUS.csv"
df = pd.read_csv(file_path)

# 2. Select Relevant Features for Clustering
df = df[['Rainfall', 'Humidity3pm', 'WindGustSpeed', 'Pressure3pm', 'Cloud3pm']].dropna()

# Standardize the features for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Determine the Optimal Number of Clusters using the Elbow Method
wcss = []
K = range(1, 11)  # Checking clusters from k=1 to k=10
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8,4))
plt.plot(K, wcss, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Determining Optimal k')
plt.show()

# From the Elbow plot, choose an optimal k (let's assume k=3 based on the elbow point)
k_optimal = 3

# 4. Initialize K-Means with the optimal number of clusters
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)

# Fit K-Means to the scaled data
kmeans.fit(X_scaled)

# Predict cluster assignments
clusters = kmeans.predict(X_scaled)

# Evaluate clustering performance using Silhouette Score
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score for k={k_optimal}: {silhouette_avg:.2f}")

# 5. Visualize the Clusters using PCA (reducing data to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=clusters,
    palette='viridis',
    alpha=0.7
)
plt.title('K-Means Clustering of Weather Data (PCA-Reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()
