import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Set seaborn style for aesthetics
sns.set(style="whitegrid")

# 1. Loading the Weather Dataset
file_path = "weatherAUS.csv"
weather = pd.read_csv(file_path)

# Drop the 'Date' column if present
if 'Date' in weather.columns:
    weather.drop(columns=['Date'], inplace=True)

# Convert categorical columns to numeric using label encoding
categorical_cols = weather.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    weather[col] = weather[col].astype(str).str.strip()
    weather[col] = label_encoder.fit_transform(weather[col])

# Handle missing values using mean imputation for numerical columns
imputer = SimpleImputer(strategy="mean")
weather.iloc[:, :] = imputer.fit_transform(weather)

# Select features for clustering
df = pd.DataFrame(weather, columns=['Rainfall', 'Humidity3pm'])
print("First five rows of the Weather dataset:")
print(df.head())

# 2. Define K-Means Function (Professor's Lab4 Implementation)
def k_means(X, k, max_iters=100):
    np.random.seed(42)
    initial_indices = np.random.choice(len(X), k, replace=False)
    centroids = X[initial_indices]

    centroids_history = [centroids.copy()]

    for i in range(max_iters):
        # Assign clusters based on closest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # Compute new centroids
        new_centroids = np.array([X[clusters == j].mean(axis=0) if len(X[clusters == j]) > 0 else centroids[j] for j in range(k)])

        if np.allclose(centroids, new_centroids):
            print(f"Converged after {i+1} iterations.")
            break

        centroids = new_centroids
        centroids_history.append(centroids.copy())

    return centroids, clusters, centroids_history

# 3. Running K-Means Clustering
X_np = df[['Rainfall', 'Humidity3pm']].values
k = 3  # Using 3 clusters as in Lab4
centroids, clusters, centroids_history = k_means(X_np, k)

print("\nFinal Centroids:")
print(centroids)

# 4. Visualizing the K-Means Learning Process
def plot_centroids_history(X, centroids_history, k):
    plt.figure(figsize=(10, 8))
    plt.title('K-Means Centroid Movements')
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Humidity at 3PM (%)')

    plt.scatter(X[:, 0], X[:, 1], s=30, c='gray', alpha=0.5, label='Data Points')

    # Colors for centroids
    centroid_colors = ['r', 'g', 'b']

    for i, centroids in enumerate(centroids_history):
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    s=200,
                    c=centroid_colors,
                    marker='X',
                    edgecolor='k',
                    label='Centroids' if i == 0 else "")
        if i > 0:
            previous_centroids = centroids_history[i - 1]
            for j in range(k):
                plt.plot([previous_centroids[j, 0], centroids[j, 0]],
                         [previous_centroids[j, 1], centroids[j, 1]],
                         'k--', linewidth=1)

    plt.legend()
    plt.show()

plot_centroids_history(X_np, centroids_history, k)

# 5. Final Clustering Visualization
def plot_final_clusters(X, clusters, centroids):
    plt.figure(figsize=(8, 6))
    plt.title('Final K-Means Clustering')
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Humidity at 3PM (%)')

    colors = ['r', 'g', 'b']

    for i in range(k):
        points = X[clusters == i]
        plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[i], label=f'Cluster {i+1}')

    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', marker='X', edgecolor='k', label='Centroids')

    plt.legend()
    plt.show()

plot_final_clusters(X_np, clusters, centroids)

# 6. Creating Animation of K-Means Process
def create_k_means_animation(X, centroids_history, k, save=False, filename='k_means_animation.gif'):
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.title('K-Means Clustering Animation')
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Humidity at 3PM (%)')

    colors = ['r', 'g', 'b']

    scatter = ax.scatter(X[:, 0], X[:, 1], s=30, c='gray', alpha=0.5, label='Data Points')

    centroids_scatter = ax.scatter([], [], s=200, c='yellow', marker='X', edgecolor='k', label='Centroids')

    ax.legend()

    def update(frame):
        ax.clear()
        plt.title('K-Means Clustering Animation')
        plt.xlabel('Rainfall (mm)')
        plt.ylabel('Humidity at 3PM (%)')

        plt.scatter(X[:, 0], X[:, 1], s=30, c='gray', alpha=0.5, label='Data Points')

        current_centroids = centroids_history[frame]
        plt.scatter(current_centroids[:, 0], current_centroids[:, 1],
                    s=200, c='yellow', marker='X', edgecolor='k', label='Centroids')

        plt.legend()

    anim = FuncAnimation(fig, update, frames=len(centroids_history), repeat=False, interval=1000)

    if save:
        anim.save(filename, writer='imagemagick')
        print(f"Animation saved as {filename}")
    else:
        plt.show()

create_k_means_animation(X_np, centroids_history, k, save=False)
