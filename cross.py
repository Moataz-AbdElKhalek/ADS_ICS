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
y = dataset['Alert']

ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)
importance = np.abs(ridge.coef_)
feature_names = X.columns
plt.bar(height=importance, x=feature_names)
plt.title("Feature importances via coefficients")
plt.show()

threshold = np.sort(importance)[-3] + 0.01

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

#change values into csv files
column_names = feature_names[sfs_forward.get_support()]
dataset_export = dataset[column_names]
dataset_export
dataset_export.to_csv("cross_validation.csv", index=False)
df = pd.read_csv("cross_validation.csv")
def code_values(m):
    if m == 0:
        return 'BENIGN'
    elif 1 in m:
        return 'Brute Force'
    elif 2 in m:
        return 'XSS'
    elif 3 in m:
        return 'Sql Injection'
df["Alert"] = df['Alert'].apply(code_values)
df.to_csv("cross_validation.csv", index=False)
