import numpy as np
from numpy import ravel
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import matplotlib.pyplot as plt

def prepareForModelUse(populatedDataFrame: DataFrame):
    
    # Uklanjanje nepotrebnih svojstava skupa podataka
    populatedDataFrame.drop('sessionIndex', axis = 1, inplace = True)
    populatedDataFrame.drop('rep', axis = 1, inplace = True)

    # Razdvajanje skupa podataka na 2 cjeline. Y oznacava skup podataka
    # koji sadrzi korisnike dok je X skup podataka kojega model strojnog
    # ucenja treba razvrstati (prepoznati) prema skupu X
    X = populatedDataFrame

    # Podijeli prethodne skupove podataka na skupove za treniranje modela 
    # strojnog ucenja i skupove za testiranje modela strojnog ucenja
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    
    trainBatchSize = 320
    testBatchSize = 80
    
    for i in range(0, len(X), trainBatchSize + testBatchSize):
        currentXBatch = X.iloc[i:i + trainBatchSize + testBatchSize]
        trainXBatch = currentXBatch.iloc[:trainBatchSize]
        testXBatch = currentXBatch.iloc[:trainBatchSize]
        
        X_train = X_train._append(trainXBatch, ignore_index = True)
        X_test = X_test._append(testXBatch, ignore_index = True)

    
    Y_train = X_train['subject']
    Y_test = X_test['subject']
    
    X_train = X_train.drop(columns=['subject'])
    X_test = X_test.drop(columns=['subject'])
    
    # Normaliziranje skupova podataka kako bi predvidanje modela 
    # bilo sto preciznije
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test

# def splitDataFrame(populatedDataFrame: DataFrame):
#     X_train = pd.DataFrame()
#     X_test = pd.DataFrame()
    
#     trainBatchSize = 320
#     testBatchSize = 80
    
#     for i in range(0, len(populatedDataFrame), trainBatchSize + testBatchSize):
#         currentXBatch = populatedDataFrame.iloc[i:i + trainBatchSize + testBatchSize]
#         trainXBatch = currentXBatch.iloc[:trainBatchSize]
#         testXBatch = currentXBatch.iloc[trainBatchSize:trainBatchSize + testBatchSize]
        
#         X_train = X_train._append(trainXBatch, ignore_index = True)
#         X_test = X_test._append(testXBatch, ignore_index = True)
        
#     return X_train, X_test

def useModel(trainDF_Y, trainDF_X, y_test, testDF_X, model: RandomForestClassifier):
    
    # Treniranje modela strojnog ucenja
    model.fit(trainDF_X, trainDF_Y.values.ravel())
    
    # Testiranje modela strojnog ucenja
    prediction = model.predict(testDF_X)
    
    # Izrada konfuzijske matrice predvidanja kako bi mogli izracunati
    # statisticke pokazatelje modela strojnog ucenja
    confusionMatrix = confusion_matrix(y_test, prediction)
    
    return confusionMatrix, prediction


def calculateStatisticalData(confusionMatrix: confusion_matrix, y_test, prediction):
    
    # Izracun statistickih pokazatelja modela strojnog ucenja
    # iz dobivene konfuzijske matrice
    truePositive = np.diag(confusionMatrix)
    falsePositive = confusionMatrix.sum(axis=0) - np.diag(confusionMatrix)
    falseNegative = confusionMatrix.sum(axis=1) - np.diag(confusionMatrix)
    trueNegative = confusionMatrix.sum() - (falsePositive + falseNegative + truePositive)
    precision = precision_score(y_test, prediction, average = 'weighted') * 100
    recall = truePositive / (truePositive + falseNegative) * 100
    falsePositiveRate = falsePositive / (falsePositive + trueNegative) * 100
    trueNegativeRate = trueNegative / (trueNegative + falsePositive) * 100
    accuracy = accuracy_score(y_test, prediction) * 100
    fMeassure = 2 * ((precision * recall) / (precision + recall))
    
    # Spremanje dobivenih statistickih podataka u rijecnik za
    # jednostavnije koristenje kasnije
    statisticalData = dict()
    statisticalData['truePositive'] = truePositive
    statisticalData['falsePositive'] = falsePositive
    statisticalData['falseNegative'] = falseNegative
    statisticalData['trueNegative'] = trueNegative
    statisticalData['precision'] = precision
    statisticalData['recall'] = recall
    statisticalData['falsePositiveRate'] = falsePositiveRate
    statisticalData['trueNegativeRate'] = trueNegativeRate
    statisticalData['accuracy'] = accuracy
    statisticalData['fMeassure'] = fMeassure
    
    return statisticalData

def plotStatisticalData(statisticalData: dict, index):
    
    # Iscrtavanje odabranih pokazatelja modela strojnog ucenja
    # Opoziv
    recallValuesDF = pd.DataFrame(list(statisticalData.get("recall")), index = index)
    recallValuesDF.plot(kind = 'bar')
    plt.xticks(rotation=90)
    plt.title('Opoziv po kategorijama')
    plt.legend().remove()
    plt.savefig('recall.png',bbox_inches='tight')
    # FPR
    falsePositiveValuesDF = pd.DataFrame(statisticalData.get("falsePositiveRate"), index = index)
    falsePositiveValuesDF.plot(kind = 'bar')
    plt.xticks(rotation=90)
    plt.title('Stopa pogrešnih klasifikacija po subjektima')
    plt.legend().remove()
    plt.savefig('fpr.png',bbox_inches='tight')
    # TNR
    trueNegativeValuesDF = pd.DataFrame(statisticalData.get("trueNegativeRate"), index = index)
    trueNegativeValuesDF.plot(kind = 'bar')
    plt.xticks(rotation=90)
    plt.title('Stopa točnih klasifikacija normalnih zapisa po subjektima')
    plt.legend().remove()
    plt.savefig('tnr.png',bbox_inches='tight')
    # F-mjera
    fMeassureValuesDF = pd.DataFrame(statisticalData.get("fMeassure"), index = index)
    fMeassureValuesDF.plot(kind = 'bar')
    plt.xticks(rotation=90)
    plt.title('F-mjera modela po subjektima')
    plt.legend().remove()
    plt.savefig('fMeassure.png',bbox_inches='tight')
    return