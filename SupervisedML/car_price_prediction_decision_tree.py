#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# estimator
from sklearn.tree import DecisionTreeRegressor
# model selection and validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.model_selection import ShuffleSplit, cross_val_score
# model metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
#%% loading the data
path = "data/car details v4.csv"
df = pd.read_csv(path)
df.info()
#%% looking at missing data
df.dropna(inplace=True) # not the best idea, you should simple imputer
null_df = df.isnull().T
sns.heatmap(null_df)
# df.dropna().shape
#%% exploring the output variable (Y Data)
sns.displot(data=df, x='Price', kind='hist')
plt.show()
sns.boxplot(data=df, x='Price')
plt.show()
df['Price2'] = np.log(df['Price']) # fixing skewness of output variable
sns.displot(data=df, x='Price2', kind='hist')
#%% breaking the data
X = df.drop(columns=['Price', 'Price2'])
y = df.Price2
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size=.2, 
                                                    random_state=0)
#%% model training and scoring
num_cols = ['Kilometer','Length','Width','Height','Fuel Tank Capacity']
cat_cols = ['Model','Fuel Type','Transmission','Location',
            'Color','Owner','Seller Type','Engine','Max Power',
            'Max Torque','Drivetrain', 'Year', 'Seating Capacity','Make']
num_pipe = Pipeline(steps=[
    ('impute', SimpleImputer()),
    ('scale', StandardScaler()),
])
cat_ord_pipe = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OrdinalEncoder()),
])
transformer = ColumnTransformer(
    transformers=[
    ('numerical', num_pipe, num_cols),
    ('categorical', cat_ord_pipe, cat_cols)]
)
#%%
model = Pipeline(steps=[
    ('transform', transformer),
    ('model', DecisionTreeRegressor(max_depth=10))
    ])
model.fit(X, y)
#%%
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred) * 100
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
#%% visualize
y_pred = model.predict(X)
temp_df = df.copy()
temp_df['pred'] = y_pred

sns.histplot(temp_df, x='Price2', bins=range(0, 250,10),
             kde=True,alpha=.1)
sns.histplot(temp_df, x='pred', bins=range(0, 250,10),
             alpha=.1, kde=True, color='red')
data = f'Multiple Linear Reg.\nScore = {score:.2f}\nMAE = {mae:.2f}\nMSE = {mse:.2f}'
plt.show()























