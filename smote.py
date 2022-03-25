import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from numpy import where
from imblearn.over_sampling import SMOTE

#df = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
#def code_values(m):
#    if m == 'BENIGN':
#        return 0
#    elif 'Brute Force' in m:
#        return 1
#    elif 'XSS' in m:
#        return 2
#    elif 'Sql Injection' in m:
#        return 3
#df["Alert"] = df[' Label'].apply(code_values)
#df.to_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", index=False)

#define dataset
dataset = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
X = dataset
X.columns = X.columns.str.strip()

X = X.loc[:, X.columns != 'Label']
X = X.loc[:, X.columns != 'Alert']
X_array = X.to_numpy()
y = dataset['Alert']
names = X.columns
y0 = dataset[dataset.Alert == 0].shape[0]
y1 = dataset[dataset.Alert == 1].shape[0]
y2 = dataset[dataset.Alert == 2].shape[0]
y3 = dataset[dataset.Alert == 3].shape[0]

#summarize class distribution
counter = Counter(y)
print(counter)

#set strategies
maj = y0/5
min = maj/2
maj = int(maj)
min = int(min)

# define pipeline
strategy = {1:min, 2:min, 3:min}
strategy2 = {0:maj}
over = SMOTE(sampling_strategy=strategy)
under = RandomUnderSampler(sampling_strategy=strategy2)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
# transform the dataset
X_array, y = pipeline.fit_resample(X_array, y)

# summarize the new class distribution
counter = Counter(y)
print(counter)

# scatter plot of examples by class label
#for label,  in counter.items():
#    row_ix = where(y == label)[0]
#    plt.scatter(X_array[row_ix, 0], X_array[row_ix, 1], X_array[row_ix, 2], X_array[row_ix, 3], label=str(label))
#plt.legend()
#plt.show()

#change values into csv files
X_df = pd.DataFrame(X_array, columns=X.columns)
y_df = pd.DataFrame(y, columns=['Alert'])
df = pd.concat([X_df, y_df], axis=1)
df.to_csv("smote_init.csv", index=False)




