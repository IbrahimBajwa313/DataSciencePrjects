import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions


ds = pd.read_csv('sales_data.csv')
# print(ds.isnull().sum())

x= ds.iloc[:,:-1]
y = ds['Purchased']

# Scaling the data as their is a large difference bw age and Estimated salary
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

sc.fit(x)
x= pd.DataFrame(sc.transform(x), columns=x.columns )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test = train_test_split(x,y, random_state=42)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
print(dt.score(x_test, y_test))

# print(dt.predict([[19,19000]]))


# Let see the Graph

from sklearn.tree import plot_tree
plt.figure(figsize=(15,10))
# plot_tree(dt)
# print(plt.show())

plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=dt)
# plt.show()

# As we know that Graphs are non linear algos, let see its scatering on the graph
sns.scatterplot( x='Age', y='EstimatedSalary', data=ds, hue='Purchased')
# plt.show()

# 
