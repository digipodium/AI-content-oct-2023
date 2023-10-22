#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
url = 'https://raw.githubusercontent.com/digipodium/Datasets/main/regression/Position_Salaries.csv'
df = pd.read_csv(url)
df.plot('Salary')
plt.show()
#%%
from sklearn.preprocessing import PolynomialFeatures
#%%
X = df[['Level']] # X should be always => 2d data
y = df['Salary']
#%%
Xp2 = PolynomialFeatures(degree=2).fit_transform(X)
Xp3 = PolynomialFeatures(degree=3).fit_transform(X)
Xp4 = PolynomialFeatures(degree=4).fit_transform(X)
Xp5 = PolynomialFeatures(degree=5).fit_transform(X)
#%%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#%%
model1, model2, model3, model4, model5 = LinearRegression(),LinearRegression(),LinearRegression(),LinearRegression(),LinearRegression()
#%%
model1.fit(X,y) # orignal
model2.fit(Xp2,y) # degree 2
model3.fit(Xp3,y) # degree 3
model4.fit(Xp4,y) # degree 4
model5.fit(Xp5,y) # degree 5
#%%
y_pred1 = model1.predict(X)
y_pred2 = model2.predict(Xp2)
y_pred3 = model3.predict(Xp3)
y_pred4 = model4.predict(Xp4)
y_pred5 = model5.predict(Xp5)
#%%
score1 = r2_score(y, y_pred1)
score2 = r2_score(y, y_pred2)
score3 = r2_score(y, y_pred3)
score4 = r2_score(y, y_pred4)
score5 = r2_score(y, y_pred5)
#%%
mae1 = mean_absolute_error(y, y_pred1)
mae2 = mean_absolute_error(y, y_pred2)
mae3 = mean_absolute_error(y, y_pred3)
mae4 = mean_absolute_error(y, y_pred4)
mae5 = mean_absolute_error(y, y_pred5)
#%%
mse1 = mean_squared_error(y, y_pred1)
mse2 = mean_squared_error(y, y_pred2)
mse3 = mean_squared_error(y, y_pred3)
mse4 = mean_squared_error(y, y_pred4)
mse5 = mean_squared_error(y, y_pred5)
#%%
xi,yi = 2,800000
fig, ((a1,a2),(a3,a4),(a5,a6)) = plt.subplots(nrows=3, 
                                              ncols=2, figsize=(10,15)) 
a1.scatter(x=X.Level, y=y)
a1.plot(X.Level,y_pred1, color='red')
#%%
a2.scatter(x=X.Level, y=y)
a2.plot(X.Level,y_pred2, color='red')

a3.scatter(x=X.Level, y=y)
a3.plot(X.Level,y_pred3, color='red')

a4.scatter(x=X.Level, y=y)
a4.plot(X.Level,y_pred4, color='red')

a5.scatter(x=X.Level, y=y)
a5.plot(X.Level,y_pred4, color='red')
#%%
data = f'Degrees 1 .\nScore = {score1:.2f}\nMAE = {mae1:.2f}\nMSE = {mse1:.2f}'
a1.text(xi,yi, data, bbox={'facecolor':'yellow','alpha':.2,'boxstyle': 'round'})
data = f'Degrees 2 .\nScore = {score2:.2f}\nMAE = {mae2:.2f}\nMSE = {mse2:.2f}'
a2.text(xi,yi, data, bbox={'facecolor':'yellow','alpha':.2,'boxstyle': 'round'})
data = f'Degrees 3 .\nScore = {score3:.2f}\nMAE = {mae3:.2f}\nMSE = {mse3:.2f}'
a3.text(xi,yi, data, bbox={'facecolor':'yellow','alpha':.2,'boxstyle': 'round'})
data = f'Degrees 4 .\nScore = {score4:.2f}\nMAE = {mae4:.2f}\nMSE = {mse4:.2f}'
a4.text(xi,yi, data, bbox={'facecolor':'yellow','alpha':.2,'boxstyle': 'round'})
data = f'Degrees 5 .\nScore = {score5:.2f}\nMAE = {mae5:.2f}\nMSE = {mse5:.2f}'
a5.text(xi,yi, data, bbox={'facecolor':'yellow','alpha':.2,'boxstyle': 'round'})
#%%
plt.savefig('polynomial_regression.png', dpi=100, bbox_inches='tight' )
#%%