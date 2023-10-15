#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
url = 'https://raw.githubusercontent.com/digipodium/Datasets/main/regression/Salary_Data.csv'
df = pd.read_csv(url)
#%% visualize
df.plot.scatter(x='YearsExperience', y='Salary',
                title='Years vs Salary')
#%%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
#%%
X = df[['YearsExperience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=0)
#%%
model = LinearRegression()
model.fit(X_train, y_train)
#%%
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred) * 100
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
#%% visualize
tdf = df.copy()
tdf['prediction'] = model.predict(X)
fig, ax = plt.subplots()
tdf.plot.scatter(x='YearsExperience', y='Salary',
                title='Years vs Salary', ax=ax)
tdf.plot.line(x='YearsExperience', y='prediction',
              ax=ax, color='red')
plt.text(x=2,y=100000, s=f'accuracy={score:.2f}%')
plt.text(x=2,y=94000, s=f'mae={mae:.2f}')
plt.show()
#%% prediction a random value
inp = pd.DataFrame({'YearsExperience': [5, 10, 15]})
result = model.predict(inp)
print(f"Approx Salary = {result}")
#%%
# THE END













