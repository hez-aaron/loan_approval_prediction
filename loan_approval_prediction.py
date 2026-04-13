import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('LoanApprovalPrediction.csv')
print(data.head())

# Get the number of columns with string data
str = (data.dtypes == 'str')
print('Categorical variables:', len(list(str[str].index)))

data.drop(['Loan_ID'], axis=1, inplace=True) # Loan_ID is not correlated with any of the other columns

str = (data.dtypes == 'str')
string_cols = list(str[str].index)
plt.figure(figsize=(9,8))
index = 1
# fig, axs = plt.subplots(3, 3, figsize=(14, 14), dpi=95)

for col in string_cols:
    y = data[col].value_counts()
    plt.subplot(2,4,index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    plt.title(col)
    index +=1
plt.tight_layout()
plt.show()
