# 1. Inicijalizacija programa #

# Uvoz potrebnih varijabli
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from functions import prepareForModelUse, useModel, calculateStatisticalData, plotStatisticalData, saveToExcel

# 2. Dohvat i priprena skupa podataka #

# Dohvat skupa podataka iz korijenske datoteke programa
currentPath = Path(__file__).parent.absolute()
datasetFilePath = Path.joinpath(currentPath, "DSL-StrongPasswordData.csv")

# Spremanje skupa podataka u "Dataframe"
datasetDataFrame = pd.read_csv(datasetFilePath)

# Spremanje liste subjekata za kasnije koristenje s 
# statistickim podacima
index = datasetDataFrame['subject'].unique()

# 3. Priprema skupa podataka #

# Koristenje napravljene metode za pripremu skupa podataka 
# za unos u model strojng ucenja
X_train, X_test, y_train, y_test = prepareForModelUse(datasetDataFrame, index)

# 4. Unos skupa podataka u model strojnog ucenja #

# Inicijalizacija modela strojnog ucenja
modelRF = RandomForestClassifier(n_estimators = 30)

# Koristenje napravljene metode za rad nad modelom strojnog ucenja
confusionMatrices, prediction, trainingTime, testingTime = useModel(y_train, X_train, y_test, X_test, modelRF)

# 5. Izračun statističkih pokazatelja modela strojnog učenja, 
# iscrtavanje grafova pojedinih pokazatelja
# i spremanje podataka u Excel datoteku #

# Metoda sluzi za izracun statisticki pokazatelja, odnosno
# sluzi za izracun performansi modela strojnog ucenja
statisticalData = calculateStatisticalData(confusionMatrices, y_test, prediction)

# Metoda kojom se iscrtavaju grafovi odabranih pokazatelja kako bi 
# smo kasnije detaljno analizirali rad modela
plotStatisticalData(statisticalData, index, confusionMatrices, modelRF)

# Metoda koja sluzi za perzistenciju podataka. Izracunate statisticke
# pokazatelje spremamo u Excel datoteku
saveToExcel(statisticalData, trainingTime, testingTime, index)