#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
#%%
iris = sns.load_dataset('iris')
#%%
sns.scatterplot(data=iris, x='petal_length', y='petal_width')
plt.show()
#%%
sns.scatterplot(data=iris, x='petal_length', y='petal_width', hue='species')
plt.show()
#%%
X = iris[['petal_length', 'petal_width']]
wcss = []
for i in range(1, 11):
    model = KMeans(n_clusters=i, n_init='auto')
    model.fit(X)
    wcss.append(model.inertia_) # wcss
    print(f"fitting {i} clusters")
#%%
plt.plot(range(1,11),wcss,'o--r')
plt.xticks(range(1,11))
plt.grid(True)
plt.show()
#%% 3 is the best numbers cluster
#%%
model = KMeans(n_clusters=3, n_init='auto')
model.fit(X)
iris['pred_species'] = model.predict(X)
#%%
sns.scatterplot(data=iris, x='petal_length', y='petal_width', hue='pred_species')
plt.show()
#%%
#%%
#%%

