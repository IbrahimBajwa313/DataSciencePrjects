import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import pandas as pd 
import seaborn as sns 

# Generate moon-shaped data
x,y = make_moons(n_samples=300, noise=0.5 )

# df = {"data1":x[:,-1 ] }
print(x[:,0 ])

# DBSCAN parameters
epsilon = 0.3
min_samples = 5