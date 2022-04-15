import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from numpy import where
from imblearn.over_sampling import SMOTE
from joblib import dump

df = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
def code_values(m):
    if m == 'BENIGN':
        return 0
    elif 'Brute Force' in m:
        return 1
    elif 'XSS' in m:
        return 2
    elif 'Sql Injection' in m:
        return 3
df["Alert"] = df[' Label'].apply(code_values)
df.to_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", index=False)

#define dataset
#dataset = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)
df.shape
df.head()
X = df
X.columns = X.columns.str.strip()

X = X.loc[:, X.columns != 'Label']
X = X.loc[:, X.columns != 'Alert']
X = X.loc[:, X.columns != 'Destination Port']
X = X.loc[:, X.columns != 'Flow Duration']
X = X.loc[:, X.columns != 'Total Fwd Packets']
X = X.loc[:, X.columns != 'Total Backward Packets']
X = X.loc[:, X.columns != 'Total Length of Fwd Packets']
X = X.loc[:, X.columns != 'Total Length of Bwd Packets']
X = X.loc[:, X.columns != 'Fwd Packet Length Max']
X = X.loc[:, X.columns != 'Fwd Packet Length Mean']
X = X.loc[:, X.columns != 'Fwd Packet Length Std']
X = X.loc[:, X.columns != 'Bwd Packet Length Max']
X = X.loc[:, X.columns != 'Flow Bytes/s']
X = X.loc[:, X.columns != 'Flow Packets/s']
X = X.loc[:, X.columns != 'Flow IAT Mean']
X = X.loc[:, X.columns != 'Flow IAT Std']
X = X.loc[:, X.columns != 'Flow IAT Max']
X = X.loc[:, X.columns != 'Flow IAT Min']
X = X.loc[:, X.columns != 'Fwd IAT Total']
X = X.loc[:, X.columns != 'Fwd IAT Mean']
X = X.loc[:, X.columns != 'Fwd IAT Std']
X = X.loc[:, X.columns != 'Fwd IAT Max']
X = X.loc[:, X.columns != 'Fwd IAT Min']
X = X.loc[:, X.columns != 'Bwd IAT Total']
X = X.loc[:, X.columns != 'Bwd IAT Mean']
X = X.loc[:, X.columns != 'Bwd IAT Std']
X = X.loc[:, X.columns != 'Bwd IAT Max']
X = X.loc[:, X.columns != 'Bwd IAT Min']
X = X.loc[:, X.columns != 'Fwd PSH Flags']
X = X.loc[:, X.columns != 'Fwd URG Flags']
X = X.loc[:, X.columns != 'Bwd URG Flags']
X = X.loc[:, X.columns != 'Bwd Header Length']
X = X.loc[:, X.columns != 'Fwd Packets/s']
X = X.loc[:, X.columns != 'Bwd Packets/s']
X = X.loc[:, X.columns != 'Min Packet Length']
X = X.loc[:, X.columns != 'Packet Length Variance']
X = X.loc[:, X.columns != 'SYN Flag Count']
X = X.loc[:, X.columns != 'RST Flag Count']
X = X.loc[:, X.columns != 'Average Packet Size']
X = X.loc[:, X.columns != 'Avg Bwd Segment Size']
X = X.loc[:, X.columns != 'Subflow Fwd Packets']
X = X.loc[:, X.columns != 'Subflow Bwd Packets']
X = X.loc[:, X.columns != 'Init_Win_bytes_forward']
X = X.loc[:, X.columns != 'act_data_pkt_fwd']
X = X.loc[:, X.columns != 'Active Mean']
X = X.loc[:, X.columns != 'Active Std']
X = X.loc[:, X.columns != 'Active Max']
X = X.loc[:, X.columns != 'Active Min']
X = X.loc[:, X.columns != 'Idle Mean']
X = X.loc[:, X.columns != 'Idle Std']
X = X.loc[:, X.columns != 'Idle Max']
X = X.loc[:, X.columns != 'Idle Min']
X_array = X.to_numpy()
y = df['Alert']
names = X.columns
#y0 = df[df.Alert == 0].shape[0]
#y1 = df[df.Alert == 1].shape[0]
#y2 = df[df.Alert == 2].shape[0]
#y3 = df[df.Alert == 3].shape[0]

#summarize class distribution
counter = Counter(y)
print(counter)
"""
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

#Testing
dataset = pd.read_csv("smote_init.csv")
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset.fillna(0, inplace=True)
dataset.shape
dataset.head()
X = dataset.drop('Alert', axis=1)
y = dataset['Alert']
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
"""
# import model from file
from joblib import load
classifier = load('thursdaymorning.joblib')
y_pred = classifier.predict(X)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))