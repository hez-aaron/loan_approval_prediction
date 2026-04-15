import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('LoanApprovalPrediction.csv')
print(data.head())

# Get the number of columns with string data
str_data = (data.dtypes == 'str')
print('Categorical variables:', len(list(str_data[str_data].index)))

# Loan_ID is not correlated with any of the other columns
data.drop(['Loan_ID'], axis=1, inplace=True) 

# Visualize categorical values in using barplot
str_data = (data.dtypes == 'str')
string_cols = list(str_data[str_data].index)
fig, axs = plt.subplots(2, 3, figsize=(9, 6), dpi=95)

for i , ax in enumerate(axs.flatten()):
    y = data[string_cols[i]].value_counts()
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y, ax=ax)
    ax.set_title(string_cols[i])

plt.tight_layout()
plt.show()
