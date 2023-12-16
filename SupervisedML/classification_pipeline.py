# -*- coding: utf-8 -*-

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# estimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#%%
df = pd.read_csv("https://raw.githubusercontent.com/digipodium/Datasets/main/classfication/Social_Network_Ads.csv")
df
#%%
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
#%%
c1 = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])
c2 = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', DecisionTreeClassifier())
    ])
c3 = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier())
    ])
c4 = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', SVC())
    ])
c5 = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier())
    ])
c6 = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', GaussianNB())
    ])
#%% logistic regression
c1.fit(X_train, y_train)
y_pred = c1.predict(X_test)
cf1 = confusion_matrix(y_test, y_pred)
cr1 = classification_report(y_test, y_pred)
#%% decision tree
c2.fit(X_train, y_train)
y_pred = c2.predict(X_test)
cf2 = confusion_matrix(y_test, y_pred)
cr2 = classification_report(y_test, y_pred)
#%% rf
c3.fit(X_train, y_train)
y_pred = c3.predict(X_test)
cf3 = confusion_matrix(y_test, y_pred)
cr3 = classification_report(y_test, y_pred)
#%% SVC
c4.fit(X_train, y_train)
y_pred = c4.predict(X_test)
cf4 = confusion_matrix(y_test, y_pred)
cr4 = classification_report(y_test, y_pred)
#%% knn
c5.fit(X_train, y_train)
y_pred = c5.predict(X_test)
cf5 = confusion_matrix(y_test, y_pred)
cr5 = classification_report(y_test, y_pred)
#%% Naive Bayes
c6.fit(X_train, y_train)
y_pred = c6.predict(X_test)
cf6 = confusion_matrix(y_test, y_pred)
cr6 = classification_report(y_test, y_pred)
#%%
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X.values, y.values, c1, legend=2 )
plt.show()
plot_decision_regions(X.values, y.values, c2, legend=2 )
plt.show()
plot_decision_regions(X.values, y.values, c3, legend=2 )
plt.show()
plot_decision_regions(X.values, y.values, c4, legend=2 )
plt.show()
plot_decision_regions(X.values, y.values, c5, legend=2 )
plt.show()
plot_decision_regions(X.values, y.values, c6, legend=2 )
plt.show()
#%% svc model is the best performing in this dataset
from joblib import dump

dump(c4, "svc_clf_social_ads.joblib")

















#%%