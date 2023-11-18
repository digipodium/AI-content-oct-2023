#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# estimator
from sklearn.ensemble import RandomForestRegressor
# model selection and validation
from sklearn.model_selection import train_test_split
# model metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
#%%
url='https://raw.githubusercontent.com/digipodium/Datasets/main/regression/house_pricing.csv'
df = pd.read_csv(url)
df.info()
#%%
X = df.drop(columns='Price')
y = df.Price
#%%
cat_cols = X.select_dtypes(include='object').columns.tolist()
print(cat_cols)
num_cols = X.select_dtypes(include='number').columns.tolist()
print(num_cols)
#%%
num_pipe = Pipeline(steps=[
    ('scale', StandardScaler()),
])
cat_ord_pipe = Pipeline(steps=[
    ('encode', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1)
    ),
])
transformer = ColumnTransformer(
    transformers=[
    ('numerical', num_pipe, num_cols),
    ('categorical', cat_ord_pipe, cat_cols)]
)
# test
transformer.fit_transform(X)
#%%
model = Pipeline(steps=[
    ('transform', transformer),
    ('model', RandomForestRegressor())
    ])
#%%
from sklearn.model_selection import GridSearchCV
#%%
params = {
    'model__n_estimators':[10, 50, 100, 250],
    'model__max_depth': [5, 10, 15, 20, 25],
    'model__criterion': ["squared_error", 
                         "absolute_error", 
                         "friedman_mse", 
                         "poisson"]
}
gridSearch = GridSearchCV(model, 
                          params,
                          n_jobs=-1,

                          verbose=3)
#%%
results = gridSearch.fit(X, y)
#%%




















