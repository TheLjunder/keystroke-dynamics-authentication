from colorama import Fore, Style
from pathlib import Path
import pandas as pd
import time
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

currentPath = Path(__file__).parent.absolute()
datasetFilePath = Path.joinpath(currentPath, "DSL-StrongPasswordData.csv")

datasetDF = pd.read_csv(datasetFilePath)
# ['subject', 'sessionIndex', 'rep', 'H.period', 'DD.period.t', 
# 'UD.period.t', 'H.t', 'DD.t.i', 'UD.t.i', 'H.i', 'DD.i.e', 
# 'UD.i.e', 'H.e', 'DD.e.five', 'UD.e.five', 'H.five', 'DD.five.Shift.r', 
# 'UD.five.Shift.r', 'H.Shift.r', 'DD.Shift.r.o', 'UD.Shift.r.o', 'H.o',
# 'DD.o.a', 'UD.o.a', 'H.a', 'DD.a.n', 'UD.a.n', 'H.n', 'DD.n.l', 'UD.n.l',
# 'H.l', 'DD.l.Return', 'UD.l.Return', 'H.Return']

datasetDF.drop('sessionIndex', axis = 1, inplace = True)
datasetDF.drop('rep', axis = 1, inplace = True)

X = datasetDF[['H.period', 'DD.period.t', 
'UD.period.t', 'H.t', 'DD.t.i', 'UD.t.i', 'H.i', 'DD.i.e', 
'UD.i.e', 'H.e', 'DD.e.five', 'UD.e.five', 'H.five', 'DD.five.Shift.r', 
'UD.five.Shift.r', 'H.Shift.r', 'DD.Shift.r.o', 'UD.Shift.r.o', 'H.o',
'DD.o.a', 'UD.o.a', 'H.a', 'DD.a.n', 'UD.a.n', 'H.n', 'DD.n.l', 'UD.n.l',
'H.l', 'DD.l.Return', 'UD.l.Return', 'H.Return']]
Y = datasetDF['subject']

X_train, X_test, y_train, y_test = train_test_split(
  X, Y, random_state=104,test_size=0.25, shuffle=True)
 
# printing out train and test sets
 
print('X_train : ')
print(X_train.head())
print(X_train.shape)
 
print('')
print('X_test : ')
print(X_test.head())
print(X_test.shape)
 
print('')
print('y_train : ')
print(y_train.head())
print(y_train.shape)
 
print('')
print('y_test : ')
print(y_test.head())
print(y_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
 
model = RandomForestClassifier().fit(X_train,y_train)
y_pred = model.predict(X_test)
print(y_pred)