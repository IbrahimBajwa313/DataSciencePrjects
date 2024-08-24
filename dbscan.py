import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import pandas as pd 
import seaborn as sns 

# Generate moon-shaped data
X, y = make_moons(n_samples=300, noise=0.1)

# Create a DataFrame from the generated data
df = pd.DataFrame(X, columns=['data1', 'data2'])

# Apply DBSCAN
epsilon = 0.2
min_samples = 5
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
df['cluster'] = dbscan.fit_predict(X)

# Plotting the results
plt.figure(figsize=(8, 6))
sns.scatterplot(x='data1', y='data2', hue='cluster', data=df, palette='viridis', s=100 )
plt.title('DBSCAN Clustering on Moon-Shaped Data')
plt.xlabel('data1')
plt.ylabel('data2')
plt.legend(title='Cluster')
plt.show()
