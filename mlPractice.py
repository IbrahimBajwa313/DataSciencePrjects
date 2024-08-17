import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('cpi.csv')
# print("Original DataFrame:")
# print(df)


x = df.drop(columns='Outcome')
y = df['Outcome']


model = LogisticRegression()
# clone_estimator=
fs = SequentialFeatureSelector(model, k_features=5, forward=True)
fs.fit_transform(x,y)

# print(fs.k_feature_names_)