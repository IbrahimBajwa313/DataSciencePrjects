import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, LeavePOut, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

# Load the dataset (assuming 'student_placement_data.csv' is available)
ds = pd.read_csv('student_placement_data.csv')
 
# Define features and target
X = ds.iloc[:, :-1]
y = ds['placed']

def perform_cross_validation(cv_method, X, y, model_type=LogisticRegression()):
  """
  Performs cross-validation using the specified method and evaluates the model.

  Args:
      cv_method: The cross-validation method (LeaveOneOut, LeavePOut, KFold, or StratifiedKFold)
      X: The feature matrix.
      y: The target variable.
      model_type: The type of model to use (defaults to LogisticRegression)

  Returns:
      A list of accuracy scores for each fold.
  """
  scores = []
  for train, test in cv_method.split(X):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]

    # Train the model
    model = model_type()
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    scores.append(score)

  return scores

# Perform cross-validation with different methods:
print("Leave-One-Out Cross-Validation:")
loo_scores = perform_cross_validation(LeaveOneOut(), X, y)
print(f"Accuracy scores: {loo_scores}")
print(f"Average accuracy: {sum(loo_scores) / len(loo_scores)}")  # Calculate average

print("\nLeave-P-Out Cross-Validation (p=2):")
lpo_scores = perform_cross_validation(LeavePOut(p=2), X, y)
print(f"Accuracy scores: {lpo_scores}")
print(f"Average accuracy: {sum(lpo_scores) / len(lpo_scores)}")  # Calculate average

print("\nK-Fold Cross-Validation (n_splits=5):")
kf_scores = perform_cross_validation(KFold(n_splits=5), X, y)
print(f"Accuracy scores: {kf_scores}")
print(f"Average accuracy: {sum(kf_scores) / len(kf_scores)}")  # Calculate average

print("\nStratified K-Fold Cross-Validation (n_splits=5):")
skf_scores = perform_cross_validation(StratifiedKFold(n_splits=5), X, y)
print(f"Accuracy scores: {skf_scores}")
print(f"Average accuracy: {sum(skf_scores) / len(skf_scores)}")  # Calculate average

# (Optional) Plot the decision regions using the best-performing model
# You can identify the best model based on the average accuracy scores.

# Train a model using all the data
lr = LogisticRegression()
lr.fit(X, y)

# Plot decision regions
plot_decision_regions(X.to_numpy(), y.to_numpy(), clf=lr)
plt.show()