import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

# Load the dataset
ds = pd.read_csv('student_placement_data.csv')

# Plot the data
sns.scatterplot(x=ds['cgpa'], y=ds['score'], data=ds, hue='placed')
plt.show()

# Define features and target
X = ds.iloc[:, :-1]
y = ds['placed']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Initialize and train the model
lr = LogisticRegression()
lr.fit(x_train, y_train)

# Evaluate the model
print(lr.score(x_test, y_test))


plot_decision_regions(X.to_numpy(), y.to_numpy(), clf=lr)
plt.show()