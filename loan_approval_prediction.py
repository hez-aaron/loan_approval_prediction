import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Get and view data
data = pd.read_csv('LoanApprovalPrediction.csv')
print(data.info())
print(data.head())

def has_string_data():
    return (data.dtypes == 'str')

def get_string_columns():
    return list(has_string_data()[has_string_data()].index)

print('Categorical variables(string data):', len(get_string_columns()))

data.drop(['Loan_ID'], axis=1, inplace=True) # Loan_ID is not correlated with any of the other columns,so we drop it

# Visualize categorical values using barplot
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

# Visualize correlations using heatmap
plt.figure(figsize=(9,6))
sns.heatmap(data.corr(), cmap='PRGn', fmt='.2f', linewidths=2, annot=True)
plt.tight_layout()
plt.show()

# Visualize the plot for Gender and Marital Status using Catplot
sns.catplot(x='Gender', y='Married', hue='Loan_Status', kind='bar', data=data)
plt.show()

# check for any missing values in the dataset
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())

print(data.isna().sum())
print()

# Split dataset for model training
X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
print('X_train shape:', X_train.shape, 'X_test shape:', X_test.shape)
print('Y_train shape:', Y_train.shape, 'X_test shape:', Y_test.shape)
print()
knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
svc = SVC()
lc = LogisticRegression(max_iter=2000)

# Making predictions on the training set
for clf in (rfc, knn, svc, lc):
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X_train, Y_train)
    Y_pred = pipe.predict(X_train)
    print('Accuracy score of', clf.__class__.__name__, '=', 100*metrics.accuracy_score(Y_train, Y_pred))
print('Test Score')
# Making predictions on the testing set
for clf in (rfc, knn, svc, lc):
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X_train, Y_train)
    pipe.score(X_test, Y_test)
    Y_pred = pipe.predict(X_test)
    print('Accuracy score of', clf.__class__.__name__, '=', 100*metrics.accuracy_score(Y_test, Y_pred))
