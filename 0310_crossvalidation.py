#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import SelectFromModel
from time import time
from sklearn.feature_selection import SequentialFeatureSelector

#define dataset
dataset = pd.read_csv("smote_init.csv")
X = dataset
X.columns = X.columns.str.strip()
X = X.loc[:, X.columns != 'Alert']
#X_array = X.to_numpy()
y = dataset['Alert']


# In[56]:



ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)
importance = np.abs(ridge.coef_)
feature_names = X.columns
plt.bar(height=importance, x=feature_names)
plt.title("Feature importances via coefficients")
plt.show()

tic_fwd = time()
sfs_forward = SequentialFeatureSelector(
                                        ridge, n_features_to_select=30, direction="forward"
                                        ).fit(X, y)
toc_fwd = time()

print(
      "Features selected by forward sequential selection: "
      f"{feature_names[sfs_forward.get_support()]}"
      )
print(f"Done in {toc_fwd - tic_fwd:.3f}s")


# In[57]:


column_names = feature_names[sfs_forward.get_support()]


# In[58]:


dataset_export


# In[59]:


dataset_export = dataset[column_names]
dataset_export = pd.concat([dataset_export,y], axis=1)
dataset_export


# In[61]:


dataset_export.to_csv("cross_validation.csv", index=False)


# In[66]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[70]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


# In[71]:


y_pred = classifier.predict(X_test)


# In[72]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:




