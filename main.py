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
from functions import prepareForModelUse

# 2. Dohvat i priprena skupa podataka #

# Dohvat skupa podataka iz korijenske datoteke programa
currentPath = Path(__file__).parent.absolute()
datasetFilePath = Path.joinpath(currentPath, "DSL-StrongPasswordData.csv")

# Spremanje skupa podataka u "Dataframe"
datasetDF = pd.read_csv(datasetFilePath)

# 3. Priprema skupa podataka #
X_train, X_test, y_train, y_test = prepareForModelUse(datasetDF)
 
model = RandomForestClassifier().fit(X_train,y_train)
y_pred = model.predict(X_test)
print(y_pred)