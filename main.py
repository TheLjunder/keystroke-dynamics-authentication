# 1. Inicijalizacija programa #

# Uvoz potrebnih varijabli
from colorama import Fore, Style
from pathlib import Path
import pandas as pd
import time
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from functions import prepareForModelUse, useModel, calculateStatisticalData, plotStatisticalData

# 2. Dohvat i priprena skupa podataka #

# Dohvat skupa podataka iz korijenske datoteke programa
currentPath = Path(__file__).parent.absolute()
datasetFilePath = Path.joinpath(currentPath, "DSL-StrongPasswordData.csv")

# Spremanje skupa podataka u "Dataframe"
datasetDF = pd.read_csv(datasetFilePath)

# Spremanje liste subjekata za kasnije koristenje s 
# statistickim podacima
index = datasetDF['subject'].unique()
print(index)

# 3. Priprema skupa podataka #

# Koristenje napravljene metode za pripremu skupa podataka 
# za unos u model strojng ucenja
X_train, X_test, Y_train, Y_test = prepareForModelUse(datasetDF)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

# 4. Unos skupa podataka u model strojnog ucenja #

# Inicijalizacija modela strojnog ucenja
modelRF = RandomForestClassifier(n_estimators = 30)

# Koristenje napravljene metode za rad nad modelom strojnog ucenja
confusionMatrix, prediction = useModel(Y_train, X_train, Y_test, X_test, modelRF)

# 5. Izracun statistickih pokazatelja modela strojnog ucenja #

# TODO Opis metode za izracun statistike
statisticalData = calculateStatisticalData(confusionMatrix, Y_test, prediction)

# TODO Opis metode za crtanje
plotStatisticalData(statisticalData, index)