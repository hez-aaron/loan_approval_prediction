import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

# Get and view data
data = pd.read_csv('LoanApprovalPrediction.csv')
print(data.head())

def get_string_data():
    return (data.dtypes == 'str')

def get_string_columns():
    return list(get_string_data()[get_string_data()].index)

print('Categorical variables:', len(get_string_columns())) # Number of columns with categorical values

data.drop(['Loan_ID'], axis=1, inplace=True) # Loan_ID is not correlated with any of the other columns,so we drop it

# Visualize categorical values in columns with barplot
string_columns = get_string_columns()
fig, axs = plt.subplots(2, 3, figsize=(8, 5), dpi=95)

for i , ax in enumerate(axs.flatten()):
    cnt = data[string_columns[i]].value_counts()
    plt.xticks(rotation=90)
    sns.barplot(x=list(cnt.index), y=cnt, ax=ax)
    ax.set_title(string_columns[i])

plt.tight_layout()
plt.show()

# Convert categorical values into numerical labels
label_encoder = preprocessing.LabelEncoder()
for col in string_columns:
    data[col] = label_encoder.fit_transform(data[col])

print('Categorical variables:', len(get_string_columns())) # Check to ensure that there are no string data columns

# Visualize correlations with heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), cmap='PRGn', fmt='.2f', linewidths=2, annot=True)
plt.tight_layout()
plt.show()
