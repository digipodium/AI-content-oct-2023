#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
#%%
url ='datasets/data.csv'
df = pd.read_csv(url)
#%%
sns.scatterplot(data=df, x='x', y='y')
plt.show()
#%% visualization
X = df[['x']]
linkage = hierarchy.linkage(X, 'ward')
plt.figure()
dn = hierarchy.dendrogram(linkage)
plt.show()
#%%
#%%
#%%
#%%
#%%
