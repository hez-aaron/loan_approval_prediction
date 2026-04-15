import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

data = pd.read_csv('LoanApprovalPrediction.csv')
print(data.head())

# Get the number of columns with string data
str_data = (data.dtypes == 'str')
print('Categorical variables:', len(list(str_data[str_data].index)))

# Loan_ID is not correlated with any of the other columns
data.drop(['Loan_ID'], axis=1, inplace=True) 

# Visualize categorical values in columns using barplot
string_data = (data.dtypes == 'str')
string_columns = list(string_data[string_data].index)
fig, axs = plt.subplots(2, 3, figsize=(9, 6), dpi=95)

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

# Check to ensure that there are no string data colums
string_data = (data.dtypes == 'str')
print('Categorical variables:', len(list(string_data[string_data].index)))
