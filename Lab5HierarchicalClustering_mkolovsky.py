# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Weather Dataset
file_path = "weatherAUS.csv"
df = pd.read_csv(file_path)

# 2. Select Relevant Features for Clustering
df = df[['Rainfall', 'Humidity3pm', 'WindGustSpeed', 'Pressure3pm', 'Cloud3pm']].dropna()

# 3. Reduce the Dataset Size to 5,000 Samples for Efficiency
df_sampled = df.sample(n=5000, random_state=42)  # Reduce dataset size
print(f"Using {df_sampled.shape[0]} samples for hierarchical clustering.")

# Standardize the features for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_sampled)

# 4. Generate the Linkage Matrix for Dendrogram
linked = linkage(X_scaled, method='ward')

# Plot the Dendrogram to Determine the Optimal Number of Clusters
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram - Hierarchical Clustering')
plt.xlabel('Sample Index')
plt.ylabel('Euclidean Distance')
plt.show()

# 5. Apply Agglomerative Clustering with Optimal k (From Dendrogram, Assume k=3)
k = 3
hc = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')  # FIXED

# Fit and Predict Cluster Assignments
clusters = hc.fit_predict(X_scaled)

# Add Cluster Assignments to the Sampled Data
df_sampled['Cluster'] = clusters

# 6. Evaluate Clustering Performance Using Silhouette Score
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"\nSilhouette Score for k={k}: {silhouette_avg:.2f}")

# 7. Visualize the Clusters Using a Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df_sampled['Humidity3pm'],
    y=df_sampled['Rainfall'],
    hue=df_sampled['Cluster'],
    palette='Set1',
    s=100,
    alpha=0.7
)
plt.title('Hierarchical Clustering of Weather Data')
plt.xlabel('Humidity at 3PM (%)')
plt.ylabel('Rainfall (mm)')
plt.legend(title='Cluster')
plt.show()

# 8. Display Cluster Statistics
print("\nCluster Statistics:")
print(df_sampled.groupby('Cluster').mean())
