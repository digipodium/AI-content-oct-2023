# -*- coding: utf-8 -*-

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
#%%
url ='datasets/data.csv'
df = pd.read_csv(url)
#%%
sns.scatterplot(data=df, x='x', y='y')
plt.show()
#%%
X = df[['x', 'y']]
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
model = MiniBatchKMeans(n_clusters=4, n_init='auto')
model.fit(X)
df['pred'] = model.predict(X)
#%%
sns.scatterplot(data=df, x='x', y='y', hue='pred')
plt.show()
#%%
#%%
#%%

