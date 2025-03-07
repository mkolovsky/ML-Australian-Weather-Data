import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Load the dataset
file_path = 'weatherAUS.csv'
data = pd.read_csv(file_path)

# Drop the 'Date' column
data = data.drop(columns=['Date'])

# Convert categorical columns to numeric using label encoding
categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = data[col].astype(str)  # Ensure all values are strings
    data[col] = label_encoder.fit_transform(data[col])

# Handle missing values: Fill numerical columns with their mean
data.fillna(data.mean(), inplace=True)

# Define the independent variables (X) and the dependent variable (y)
X = data.drop(columns=['RainTomorrow'])
y = data['RainTomorrow']

# Add a constant to the independent variables
X = sm.add_constant(X)

# Fit the OLS regression model
model = sm.OLS(y, X).fit()

# 1. Model Summary
print("\nModel Summary:")
print(model.summary())

# 2. Residuals vs Fitted Values
fitted_vals = model.predict(X)
residuals = y - fitted_vals
plt.figure(figsize=(10, 6))
sns.scatterplot(x=fitted_vals, y=residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()

# 3. Durbin-Watson Statistic
dw_stat = durbin_watson(model.resid)
print(f"\nDurbin-Watson statistic: {dw_stat}")

# 4. Breusch-Pagan Test
bp_test = het_breuschpagan(model.resid, model.model.exog)
bp_results = dict(zip(['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'], bp_test))
print("\nBreusch-Pagan test results:")
for key, value in bp_results.items():
    print(f"{key}: {value}")

# 5. Normality of Errors: Q-Q Plot and Shapiro-Wilk Test
sm.qqplot(residuals, line='45', fit=True)
plt.title('Q-Q Plot of Residuals')
plt.show()

residuals_sample = residuals.sample(5000, random_state=1)
shapiro_test = shapiro(residuals_sample)
print(f"\nShapiro-Wilk test statistic: {shapiro_test.statistic}, p-value: {shapiro_test.pvalue}")

# 6. Variance Inflation Factor (VIF)
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factor (VIF) for each feature:")
print(vif_data)
