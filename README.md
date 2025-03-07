## Overview
This repository contains Python scripts for analyzing and modeling weather data from the **Australian Weather Dataset (weatherAUS.csv)**. The analyses include exploratory data analysis (EDA), regression modeling, clustering, classification, and association rule mining using machine learning techniques.

## Dataset
The dataset used in this project is **weatherAUS.csv**, which contains historical weather observations from multiple locations across Australia. The primary goal is to analyze weather patterns and predict **RainTomorrow** (whether it will rain the next day).

## Installation & Dependencies
Before running the scripts, install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels mlxtend
```

## Files & Descriptions

### 1. Regression Analysis
- **Lab1AUScode_mkolovsky.py**
  - Performs **Ordinary Least Squares (OLS) regression** to analyze the relationship between weather variables and **RainTomorrow**.
  - Checks for multicollinearity using **Variance Inflation Factor (VIF)**.
  - Conducts diagnostic tests such as **Durbin-Watson**, **Breusch-Pagan**, and **Shapiro-Wilk**.
  
- **Lab2-1_mkolovsky.py & Lab2AUScode_mkolovsky.py**
  - Compares **GLS, Ridge, Lasso, and Elastic Net** regression models.
  - Uses **log transformations** to handle skewed data.
  - Implements **grid search cross-validation** to optimize model parameters.
  - Evaluates models based on **Mean Squared Error (MSE)** and **R² score**.

### 2. Clustering
- **Lab4ClusteringAUS_mkolovsky.py**
  - Implements **K-Means clustering** on weather features such as **Rainfall** and **Humidity3pm**.
  - Visualizes **centroid movements** across iterations.
  - Creates an **animation** to illustrate the clustering process.

- **Lab5ClusteringKNN_mkolovsky.py**
  - Uses **K-Means clustering** with **Elbow Method** to determine the optimal number of clusters.
  - Applies **PCA (Principal Component Analysis)** to visualize clusters in 2D.
  - Evaluates clusters using **Silhouette Score**.

- **Lab5HierarchicalClustering_mkolovsky.py**
  - Implements **Agglomerative Hierarchical Clustering**.
  - Uses a **Dendrogram** to determine the best number of clusters.
  - Evaluates clustering performance with **Silhouette Score**.

### 3. Classification
- **Lab5DecisionTree_mkolovsky.py**
  - Trains a **Decision Tree Classifier** to predict **RainTomorrow**.
  - Visualizes the **decision tree**.
  - Evaluates model performance using **accuracy, confusion matrix, and classification report**.

- **Lab5NaiveBayesian_mkolovsky.py**
  - Implements a **Gaussian Naïve Bayes classifier** to predict rainfall.
  - Displays **confusion matrix** and **classification metrics**.

- **Lab5SVM_mkolovsky.py**
  - Uses **Support Vector Machine (SVM)** with a **linear kernel**.
  - Standardizes features using **StandardScaler**.
  - Evaluates model using **accuracy and confusion matrix**.

### 4. Association Rule Mining
- **Lab5Association_mkolovsky.py**
  - Applies **Apriori algorithm** to find frequent weather condition sets.
  - Generates **association rules** based on confidence and lift.
  - Visualizes association rules in a scatter plot.

## Usage
1. Ensure that **weatherAUS.csv** is available in the working directory.
2. Run any script using:
   ```bash
   python script_name.py
   ```
3. The outputs include model summaries, visualizations, and performance metrics.

## License
This project is for educational and research purposes. No warranty is provided.

## Author
M. Kolovsky

