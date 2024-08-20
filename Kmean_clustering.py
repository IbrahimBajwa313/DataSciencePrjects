import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ds = pd.read_csv("kmeans_dataset.csv")

sns.pairplot(ds)
plt.show()