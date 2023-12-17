# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 14:53:15 2023

@author: ZAID
"""
#%%
import tensorflow as tf
print(tf.__version__)
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
url = 'https://raw.githubusercontent.com/digipodium/Datasets/main/regression/bike_rental.csv'
df = pd.read_csv(url, parse_dates=['dteday'], index_col=0)
#%%
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
#%%
X = df.drop(columns=['cnt','dteday'])
X.isnull().sum()
#%%
y = df['cnt'].astype(float)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)
#%%
X_train.info()
#%%
pca = PCA(n_components=2)
Xp_train = pca.fit_transform(X_train)
Xp_test = pca.transform(X_test)
#%% Neural network architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(
    units = 5, activation = 'relu', input_dim=Xp_train.shape[1]    
))
model.add(tf.keras.layers.Dense(
    units = 6, activation = 'relu'    
))
model.add(tf.keras.layers.Dense(
    units = 1
))
model.summary()
#%%
model.compile(
    optimizer = 'adam',
    loss = 'mean_absolute_error',
    metrics = tf.keras.metrics.R2Score(),
    # run_eagerly=True
)
#%%
history = model.fit(Xp_train, 
                    y_train,
                    batch_size=8,
                    epochs=10,
                    validation_split=.2,
                    verbose=1
                    )
#%% visualiation
histdf = pd.DataFrame(history.history)
#%%
histdf[['loss','val_loss']].plot(style='o--')
#%%
histdf[['r2_score','val_r2_score']].plot(style='o--')
#%%
y_pred = model.predict(Xp_test)
#%%
fig, ax = plt.subplots(figsize=(10,6))
sns.histplot(y_test, kde=True, ax=ax, color='red')
sns.histplot(y_pred, kde=True, ax=ax)
plt.show()