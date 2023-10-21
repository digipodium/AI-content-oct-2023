#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
url = 'https://raw.githubusercontent.com/digipodium/Datasets/main/regression/oil_dataset.csv'
df = pd.read_csv(url)
#%%
sns.regplot(df, x='Mendacium', y='Price',
            line_kws=dict(color='red', linewidth=5))
plt.show()
#%%
sns.regplot(df, x='Depth', y='Price',
            line_kws=dict(color='red', linewidth=5))
plt.show()
#%%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
#%%
X, y = df.drop(columns=['Price']) , df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, 
                                                    random_state=0)
#%%
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred) * 100
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
#%% saving the model
from joblib import dump
import os

if not os.path.exists('saved_models'):
    os.makedirs('saved_models')
dump(model, 'saved_models/oil_price_model.pk') # joblib
print("model saved")
#%% visualization
y_pred = model.predict(X)
temp_df = df.copy()
temp_df['pred'] = y_pred

sns.histplot(temp_df, x='Price', bins=range(0, 250,10),
             kde=True,alpha=.1)
sns.histplot(temp_df, x='pred', bins=range(0, 250,10),
             alpha=.1, kde=True, color='red')
data = f'Multiple Linear Reg.\nScore = {score:.2f}\nMAE = {mae:.2f}\nMSE = {mse:.2f}'
plt.text(160,100, data, bbox={'facecolor':'yellow',
                              'alpha':.2,
                              'boxstyle': 'round'})
plt.show()
#%% The end











