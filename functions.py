import numpy as np
from numpy import ravel
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import matplotlib.pyplot as plt

def prepareForModelUse(populatedDataframe: DataFrame):
    
    # Uklanjanje nepotrebnih svojstava skupa podataka
    populatedDataframe.drop('sessionIndex', axis = 1, inplace = True)
    populatedDataframe.drop('rep', axis = 1, inplace = True)

    # Razdvajanje skupa podataka na 2 cjeline. Y oznacava skup podataka
    # koji sadrzi korisnike dok je X skup podataka kojega model strojnog
    # ucenja treba razvrstati (prepoznati) prema skupu X
    X = populatedDataframe[['H.period', 'DD.period.t', 
    'UD.period.t', 'H.t', 'DD.t.i', 'UD.t.i', 'H.i', 'DD.i.e', 
    'UD.i.e', 'H.e', 'DD.e.five', 'UD.e.five', 'H.five', 'DD.five.Shift.r', 
    'UD.five.Shift.r', 'H.Shift.r', 'DD.Shift.r.o', 'UD.Shift.r.o', 'H.o',
    'DD.o.a', 'UD.o.a', 'H.a', 'DD.a.n', 'UD.a.n', 'H.n', 'DD.n.l', 'UD.n.l',
    'H.l', 'DD.l.Return', 'UD.l.Return', 'H.Return']]
    Y = populatedDataframe['subject']

    # Podijeli prethodne skupove podataka na skupove za treniranje modela 
    # strojnog ucenja i skupove za testiranje modela strojnog ucenja
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, random_state=104,test_size=0.25, shuffle=True)

    # Normaliziranje skupova podataka kako bi predvidanje modela 
    # bilo sto preciznije
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def useModel(trainDF_Y, trainDF_X, y_test, testDF_X, model: RandomForestClassifier):
    model.fit(trainDF_X, trainDF_Y.values.ravel())
    prediction = model.predict(testDF_X)
    confusionMatrix = confusion_matrix(y_test, prediction)
    return confusionMatrix, prediction

def calculateStatisticalData(confusionMatrix: confusion_matrix, y_test, prediction):
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
    # # Preciznost
    # precisonValuesDF = pd.DataFrame(list(statisticalData.get("precision")), index = ['Vrijednosti pokazatelja'])
    # precisonValuesDF.plot(kind = 'bar')
    # plt.xticks(rotation=0)
    # plt.title('Preciznost pojedinog modela')
    # plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
    # plt.savefig('precision.png',bbox_inches='tight')
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
    # # Točnost
    # accuracyValuesDF = pd.DataFrame(statisticalData.get("accuracy"), index = ['Vrijednosti pokazatelja'])
    # accuracyValuesDF.plot(kind = 'bar')
    # plt.xticks(rotation=0)
    # plt.title('Točnost modela')
    # plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
    # plt.savefig('accuracy.png',bbox_inches='tight')
    # F-mjera
    fMeassureValuesDF = pd.DataFrame(statisticalData.get("fMeassure"), index = index)
    fMeassureValuesDF.plot(kind = 'bar')
    plt.xticks(rotation=90)
    plt.title('F-mjera modela po subjektima')
    plt.legend().remove()
    plt.savefig('fMeassure.png',bbox_inches='tight')
    return