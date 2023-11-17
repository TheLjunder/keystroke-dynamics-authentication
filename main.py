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
from functions import prepareForModelUse, useModel

# 2. Dohvat i priprena skupa podataka #

# Dohvat skupa podataka iz korijenske datoteke programa
currentPath = Path(__file__).parent.absolute()
datasetFilePath = Path.joinpath(currentPath, "DSL-StrongPasswordData.csv")

# Spremanje skupa podataka u "Dataframe"
datasetDF = pd.read_csv(datasetFilePath)

# 3. Priprema skupa podataka #

# Koristenje napravljene metode za pripremu skupa podataka 
# za unos u model strojng ucenja
X_train, X_test, y_train, y_test = prepareForModelUse(datasetDF)

# 4. Unos skupa podataka u model strojnog ucenja #

# Inicijalizacija modela strojnog ucenja
modelRF = RandomForestClassifier(n_estimators = 30)

# Koristenje napravljene metode za rad nad modelom strojnog ucenja
confusionMatrix = useModel(y_train, X_train, y_test, X_test, modelRF)
