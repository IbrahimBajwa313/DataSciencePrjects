# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns 

# ds= pd.read_csv('age_experience_salary.csv')    
# # print(ds.head())

# # sns.pairplot(data=ds)
# # plt.show()

# x = ds.iloc[:,:-1]
# y = ds['Salary']

# from sklearn.model_selection import train_test_split, GridSearchCV
# x_train , x_test, y_train , y_test = train_test_split(x,y , random_state=42)


# from sklearn.tree import DecisionTreeRegressor,plot_tree

# dt = DecisionTreeRegressor()
# dt.fit(x_train,y_train)

# print(dt.score(x_train,y_train))

# plot_tree(dt)
# plt.show()

# # Applying Grid SearchCV
# param_grid = {
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 10, 20],
#     'min_samples_leaf': [1, 5, 10],
#     'max_features': [None, 'auto', 'sqrt', 'log2']
# }

# grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
# grid_search.fit(x_train,y_train)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Load dataset
ds = pd.read_csv('age_experience_salary.csv')

# Define features and target variable
x = ds.iloc[:, :-1]
y = ds['Salary']

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# Initialize the DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(x_train, y_train)

# Print training score before applying GridSearchCV
print(f"Training Score (before GridSearchCV): {dt.score(x_train, y_train):.4f}")

# Visualize the decision tree before tuning
plt.figure(figsize=(12, 8))
plot_tree(dt, filled=True, feature_names=x.columns)
plt.show()

# Define the hyperparameters grid
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'max_features': [None, 'auto', 'sqrt', 'log2']
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')

# Fit the model with GridSearchCV
grid_search.fit(x_train, y_train)

# Print the best parameters and the best score from GridSearchCV
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Train the final model with the best parameters
best_dt = grid_search.best_estimator_
best_dt.fit(x_train, y_train)

# Evaluate the final model
print(f"Training Score (after GridSearchCV): {best_dt.score(x_train, y_train):.4f}")
print(f"Test Score: {best_dt.score(x_test, y_test):.4f}")

# Visualize the decision tree after tuning
plt.figure(figsize=(12, 8))
plot_tree(best_dt, filled=True, feature_names=x.columns)
plt.show()
