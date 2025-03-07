import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Weather Dataset
file_path = "weatherAUS.csv"
df = pd.read_csv(file_path)

# 2. Select Relevant Categorical Columns for Association Analysis
categorical_features = ['RainToday', 'RainTomorrow', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'Cloud9am', 'Cloud3pm']
df_filtered = df[categorical_features].dropna()  # Drop missing values

# Convert 'RainToday' and 'RainTomorrow' to Yes/No strings
df_filtered['RainToday'] = df_filtered['RainToday'].replace({1: 'Yes', 0: 'No'})
df_filtered['RainTomorrow'] = df_filtered['RainTomorrow'].replace({1: 'Yes', 0: 'No'})

# Convert DataFrame into a list of transactions
transactions = df_filtered.applymap(str).values.tolist()

# 3. One-Hot Encode the Transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print("One-Hot Encoded Transaction Data:")
print(df_encoded.head())

# 4. Apply Apriori Algorithm to Find Frequent Weather Condition Sets
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)

print("\nFrequent Weather Condition Sets:")
print(frequent_itemsets)

# 5. Generate Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# 6. Visualize the Association Rules
plt.figure(figsize=(8,6))
sns.scatterplot(
    x='support',
    y='confidence',
    size='lift',
    data=rules,
    hue='lift',
    palette='viridis',
    sizes=(100, 1000),
    alpha=0.7
)
plt.title('Association Rules Scatter Plot')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.legend(title='Lift', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 7. Display Top Association Rules Sorted by Lift
sorted_rules = rules.sort_values(by='lift', ascending=False)
print("\nTop Weather Association Rules Sorted by Lift:")
print(sorted_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
