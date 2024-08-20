import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

ds = pd.read_csv("kmeans_dataset.csv")

# Calculate WCSS (Within-Cluster Sum of Squares) for different cluster counts
wcss = []
for i in range(2, 21):
    km = KMeans(n_clusters=i, init='k-means++')
    km.fit(ds)
    wcss.append(km.inertia_)

# Plot the Elbow method graph
plt.plot([i for i in range(2, 21)], wcss, marker='o')
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.xticks([i for i in range(2, 21)])
plt.grid(axis='x')
plt.show()

# Apply KMeans with the chosen number of clusters
knn = KMeans(n_clusters=3)  # Corrected: '3' -> 3
ds['Predict'] = knn.fit_predict(ds)

# print(ds)

sns.pairplot(ds, hue = 'Predict')
plt.show()