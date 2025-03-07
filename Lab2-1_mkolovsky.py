import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import shapiro
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seaborn style
sns.set(style="whitegrid")

# Load Dataset
file_path = 'weatherAUS.csv'
data = pd.read_csv(file_path)

# Drop the 'Date' column
data = data.drop(columns=['Date'])

# Convert categorical columns to numeric using label encoding
categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = data[col].astype(str)
    data[col] = label_encoder.fit_transform(data[col])

# Handle missing values: Fill numerical columns with their median
data.fillna(data.median(), inplace=True)

# Drop highly collinear features: Pressure9am and MaxTemp
data = data.drop(columns=['Pressure9am', 'MaxTemp'])

# Feature Transformation (Log Transformation for skewed variables, ensuring no negatives)
skewed_features = ['Rainfall', 'Evaporation', 'WindGustSpeed', 'Humidity3pm', 'Temp3pm']
for feature in skewed_features:
    data[feature] = np.where(data[feature] < 0, 0, data[feature])  # Replace negatives with 0
    data[feature] = np.log1p(data[feature])  # log(1 + x) transformation

# Feature Scaling
scaler = StandardScaler()
X = data.drop(columns=['RainTomorrow'])
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
y = np.log1p(data['RainTomorrow'])  # Log transform target variable to handle heteroskedasticity

# Handling Multicollinearity
X_with_const = sm.add_constant(X_scaled)
vif_data = pd.DataFrame()
vif_data['Feature'] = X_with_const.columns
vif_data['VIF'] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
print("\nVariance Inflation Factor (VIF) for Each Feature:")
print(vif_data)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training and Selection (Using GLS instead of OLS for heteroskedasticity correction)
models = {
    'GLS': sm.GLS(y_train, sm.add_constant(X_train)).fit(),  # GLS instead of OLS
    'Ridge': GridSearchCV(Ridge(), {'alpha': [0.1, 1.0, 10.0, 100.0]}, cv=5, scoring='neg_mean_squared_error').fit(X_train, y_train),
    'Lasso': GridSearchCV(Lasso(max_iter=10000), {'alpha': [0.01, 0.1, 1.0, 10.0]}, cv=5, scoring='neg_mean_squared_error').fit(X_train, y_train),
    'ElasticNet': GridSearchCV(ElasticNet(max_iter=10000), {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]}, cv=5, scoring='neg_mean_squared_error').fit(X_train, y_train)
}

# Model Evaluation
model_comparison = pd.DataFrame({
    'Model': ['GLS', 'Ridge', 'Lasso', 'Elastic Net'],
    'MSE': [
        mean_squared_error(y_test, models['GLS'].predict(sm.add_constant(X_test))),
        mean_squared_error(y_test, models['Ridge'].best_estimator_.predict(X_test)),
        mean_squared_error(y_test, models['Lasso'].best_estimator_.predict(X_test)),
        mean_squared_error(y_test, models['ElasticNet'].best_estimator_.predict(X_test))
    ],
    'R²': [
        r2_score(y_test, models['GLS'].predict(sm.add_constant(X_test))),
        r2_score(y_test, models['Ridge'].best_estimator_.predict(X_test)),
        r2_score(y_test, models['Lasso'].best_estimator_.predict(X_test)),
        r2_score(y_test, models['ElasticNet'].best_estimator_.predict(X_test))
    ]
})
print("\nModel Comparison:")
print(model_comparison)

# Diagnostics on the Best Model (Assuming Ridge Performs Best)
best_model = models['Ridge'].best_estimator_
y_pred_best = best_model.predict(X_test)
residuals_best = y_test - y_pred_best

# Residuals vs Fitted Values
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_pred_best, y=residuals_best, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values (Ridge)')
plt.show()

# Durbin-Watson Test
dw_best = durbin_watson(residuals_best)
print(f'\nDurbin-Watson statistic for Ridge: {dw_best:.4f}')

# Breusch-Pagan Test
bp_test_best = het_breuschpagan(residuals_best, sm.add_constant(X_test))
print('\nBreusch-Pagan Test Results for Ridge:')
print(f'Lagrange multiplier statistic: {bp_test_best[0]:.4f}')
print(f'p-value: {bp_test_best[1]:.4f}')

# Q-Q Plot and Shapiro-Wilk Test
sm.qqplot(residuals_best, line='45', fit=True)
plt.title('Q-Q Plot of Residuals (Ridge)')
plt.show()

# Variance Inflation Factor (VIF)
vif_data_best = pd.DataFrame()
vif_data_best['Feature'] = X_with_const.columns
vif_data_best['VIF'] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
print('\nVariance Inflation Factor (VIF) for Each Feature:')
print(vif_data_best)
