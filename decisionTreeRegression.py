import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

ds= pd.read_csv('age_experience_salary.csv')    
# print(ds.head())

# sns.pairplot(data=ds)
# plt.show()

x = ds.iloc[:,:-1]
y = ds['Salary']

from sklearn.model_selection import train_test_split
x_train , x_test, y_train , y_test = train_test_split(x,y , random_state=42)

from sklearn.tree import DecisionTreeRegressor,plot_tree

dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)

print(dt.score(x_train,y_train))

plot_tree(dt)
plt.show()