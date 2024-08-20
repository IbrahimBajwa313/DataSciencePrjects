import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ds = pd.read_csv("kmeans_dataset.csv")

# sns.pairplot(ds)
# plt.show()

from sklearn.cluster import KMeans

wcss = []

for i in range( 2,21 ):
    km = KMeans( n_clusters = i , init= 'k-means++')
    km.fit(ds)
    wcss.append(km.inertia_)



plt.plot([i for i in range(2,21)],wcss,marker='o')
plt.xlabel("No. of Cluster")
plt.ylabel("WCSS")
plt.xticks([i for i in range(2,21)])
plt.grid(axis='x' )
plt.show()

knn = KMeans(n_clusters = '3')
ds['Predict']= knn.fit_predict(ds)

print(ds)
