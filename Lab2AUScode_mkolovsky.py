import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For regression modeling and diagnostics
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan

# For regularization techniques
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# For normality test
from scipy.stats import shapiro

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style
sns.set(style="whitegrid")

# 1. Load the Dataset
file_path = 'weatherAUS.csv'
data = pd.read_csv(file_path)

# Drop the 'Date' column
data = data.drop(columns=['Date'])

# 2. Exploratory Data Analysis (EDA)

# Convert categorical columns to numeric using label encoding
categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = data[col].astype(str)  # Ensure all values are strings
    data[col] = label_encoder.fit_transform(data[col])

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Handle missing values: Fill numerical columns with their mean
data.fillna(data.mean(), inplace=True)

# Statistical summary
print("\nStatistical summary of the dataset:")
print(data.describe())

# Correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# 3. Data Preprocessing

# 3.1 Feature Scaling
scaler = StandardScaler()
X = data.drop(columns=['RainTomorrow'])
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
y = data['RainTomorrow']

# 3.2 Handling Multicollinearity
X_with_const = sm.add_constant(X_scaled)
vif_data = pd.DataFrame()
vif_data['Feature'] = X_with_const.columns
vif_data['VIF'] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
print("\nVariance Inflation Factor (VIF) for Each Feature:")
print(vif_data)

# 4. Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Modeling

# 5.1 Ordinary Least Squares (OLS) Regression
X_train_ols = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_ols).fit()
print("\nOLS Regression Model Summary:")
print(ols_model.summary())

# 5.2 Ridge Regression
ridge = Ridge()
parameters_ridge = {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
ridge_reg = GridSearchCV(ridge, parameters_ridge, scoring='neg_mean_squared_error', cv=5)
ridge_reg.fit(X_train, y_train)
print("\nRidge Regression Best Parameters:")
print(ridge_reg.best_params_)

# 5.3 Lasso Regression
lasso = Lasso(max_iter=10000)
parameters_lasso = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
lasso_reg = GridSearchCV(lasso, parameters_lasso, scoring='neg_mean_squared_error', cv=5)
lasso_reg.fit(X_train, y_train)
print("\nLasso Regression Best Parameters:")
print(lasso_reg.best_params_)

# 5.4 Elastic Net Regression
elastic = ElasticNet(max_iter=10000)
parameters_elastic = {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]}
elastic_reg = GridSearchCV(elastic, parameters_elastic, scoring='neg_mean_squared_error', cv=5)
elastic_reg.fit(X_train, y_train)
print("\nElastic Net Regression Best Parameters:")
print(elastic_reg.best_params_)

# 6. Model Evaluation and Comparison

model_comparison = pd.DataFrame({
    'Model': ['OLS', 'Ridge', 'Lasso', 'Elastic Net'],
    'MSE': [
        mean_squared_error(y_test, ols_model.predict(sm.add_constant(X_test))),
        mean_squared_error(y_test, ridge_reg.best_estimator_.predict(X_test)),
        mean_squared_error(y_test, lasso_reg.best_estimator_.predict(X_test)),
        mean_squared_error(y_test, elastic_reg.best_estimator_.predict(X_test))
    ],
    'R²': [
        r2_score(y_test, ols_model.predict(sm.add_constant(X_test))),
        r2_score(y_test, ridge_reg.best_estimator_.predict(X_test)),
        r2_score(y_test, lasso_reg.best_estimator_.predict(X_test)),
        r2_score(y_test, elastic_reg.best_estimator_.predict(X_test))
    ]
})

print("\nModel Comparison:")
print(model_comparison)

# 7. Regression Diagnostics for the Best Model (Assuming Elastic Net is Best)

# 7.1. Residuals vs Fitted Values Plot
y_pred_best = elastic_reg.best_estimator_.predict(X_test)
residuals_best = y_test - y_pred_best

plt.figure(figsize=(10,6))
sns.scatterplot(x=y_pred_best, y=residuals_best, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values (Elastic Net)')
plt.show()

# 7.2. Durbin-Watson Test
dw_best = durbin_watson(residuals_best)
print(f'\nDurbin-Watson statistic for Elastic Net: {dw_best:.4f}')

# 7.3. Breusch-Pagan Test
bp_test_best = het_breuschpagan(residuals_best, sm.add_constant(X_test))
print('\nBreusch-Pagan Test Results for Elastic Net:')
print(f'Lagrange multiplier statistic: {bp_test_best[0]:.4f}')
print(f'p-value: {bp_test_best[1]:.4f}')

# 7.4. Q-Q Plot and Shapiro-Wilk Test
sm.qqplot(residuals_best, line='45', fit=True)
plt.title('Q-Q Plot of Residuals (Elastic Net)')
plt.show()

# 7.5. Variance Inflation Factor (VIF)
vif_data_best = pd.DataFrame()
vif_data_best['Feature'] = X_with_const.columns
vif_data_best['VIF'] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
print('\nVariance Inflation Factor (VIF) for Each Feature:')
print(vif_data_best)
