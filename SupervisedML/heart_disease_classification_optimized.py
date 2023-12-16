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
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import KernelPCA
#%%
url = "https://raw.githubusercontent.com/digipodium/Datasets/main/classfication/diabetes.csv"
df = pd.read_csv(url)
#%%
sns.countplot(data=df, x='Outcome')

X = df.drop(columns='Outcome')
#%% decompose into 2 column using PCA
pca = KernelPCA(n_components=2)
X = pca.fit_transform(X)
y = df['Outcome']
#%% undersample
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()
# Xu is X undersampled
Xu, yu = rus.fit_resample(X, y)

#%% oversample
from imblearn.over_sampling import SMOTE
smote = SMOTE()
# Xu is X undersampled
Xo, yo = smote.fit_resample(X, y)

#%%
print("Outcome count in Oversampled Data")
print(yo.value_counts())
print("Outcome count in Undersampled Data")
print(yu.value_counts())
#%% split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
Xu_train, Xu_test, yu_train, yu_test = train_test_split(Xu, yu, test_size=.2)
Xo_train, Xo_test, yo_train, yo_test = train_test_split(Xo, yo, test_size=.2)
#%% testing imbalance data
clf_n = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier())
        ])
clf_n.fit(X_train, y_train)
y_pred = clf_n.predict(X_test)
cfn = confusion_matrix(y_test, y_pred)
crn = classification_report(y_test, y_pred)
#%% testing undersampled data
clf_u = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier())
        ])
clf_u.fit(Xu_train, yu_train)
yu_pred = clf_u.predict(Xu_test)
cfu = confusion_matrix(yu_test, yu_pred)
cru = classification_report(yu_test, yu_pred)

#%%
clf_o = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier())
        ])
clf_o.fit(Xo_train, yo_train)
yo_pred = clf_o.predict(Xo_test)
cfo = confusion_matrix(yo_test, yo_pred)
cro = classification_report(yo_test, yo_pred)
#%%
plot_decision_regions(X, y.values, clf_n, )
plt.show()
#%%

#%%
#%%

#%%

#%%

#%%
#%%

#%%

#%%

#%%
#%%

#%%

#%%

#%%
