import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

def prepareForModelUse(populatedDataframe: DataFrame, index):
    
    # Uklanjanje nepotrebnih svojstava skupa podataka
    populatedDataframe.drop('sessionIndex', axis = 1, inplace = True)
    populatedDataframe.drop('rep', axis = 1, inplace = True)

    # Razdvajanje skupa podataka na 2 cjeline. Y oznacava skup podataka
    # koji sadrzi korisnike dok je X skup podataka kojega model strojnog
    # ucenja treba razvrstati (prepoznati) prema skupu Y
    X = populatedDataframe.iloc[:, 1:]
    Y = populatedDataframe['subject']

    # Priprema polja za skupove treniranja i testiranja. Pocetna inicijalizacija
    # je potrebna zbog zadrzavanja originalnog poredka subjekata
    X_train, X_test, y_train, y_test = [], [], [], []

    # Podijeli prethodne skupove podataka na skupove za treniranje modela 
    # strojnog ucenja i skupove za testiranje modela strojnog ucenja
    for subject in index:
        subjectData = populatedDataframe[populatedDataframe['subject'] == subject]
        trainData, testData = train_test_split(subjectData, test_size=0.2, random_state=42)
        
        X_train.append(trainData.iloc[:, 1:])
        y_train.extend(trainData['subject'])
        
        X_test.append(testData.iloc[:, 1:])
        y_test.extend(testData['subject'])

    # Dodavanje podataka u liste
    X_train = pd.concat(X_train, axis=0)
    X_test = pd.concat(X_test, axis=0)

    # Pretvorba popunjenih lista u 'DataFrame'
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    # Normaliziranje skupova podataka kako bi predvidanje modela 
    # bilo sto preciznije
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def useModel(trainDF_Y, trainDF_X, y_test, testDF_X, model: RandomForestClassifier):
    
    # Treniranje modela stojnog ucenja i mjerenje 
    # vremena potrebnog za isto
    startTime = time.time()
    model.fit(trainDF_X, trainDF_Y.values.ravel())
    endTime = time.time()
    trainingTime = endTime - startTime
    
    # Testiranje modela strojnog ucenja i mjerenje
    # vremena potrebnog za isto
    startTime = time.time()
    prediction = model.predict(testDF_X)
    endTime = time.time()
    testingTime = endTime - startTime
    
    # Izrada konfuzijske matrice klasifikacija modela. Sluzi nam za kasnije
    # izracunavanje konkretnih pokazatelja performansi modela strojnog ucenja
    confusionMatrices = confusion_matrix(y_test, prediction, labels = model.classes_)
    return confusionMatrices, prediction, trainingTime, testingTime

def calculateStatisticalData(confusionMatrices, y_test, prediction):
    
    # Izracun pokazatelja performansi modela strojnog ucenja
    truePositive = np.diag(confusionMatrices)
    falsePositive = confusionMatrices.sum(axis=0) - np.diag(confusionMatrices)
    falseNegative = confusionMatrices.sum(axis=1) - np.diag(confusionMatrices)
    trueNegative = confusionMatrices.sum() - (falsePositive + falseNegative + truePositive)
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

def plotStatisticalData(statisticalData: dict, index, confusionMatrices, model):
    
    # Iscrtavanje slozenih pokazatelja kako bi mogli usporediti performanse 
    # modela strojnog ucenja za pojedine subjekte iz skupa podataka
    # Opoziv
    recallValuesDF = pd.DataFrame(list(statisticalData.get("recall")), index = index)
    recallValuesDF.plot(kind = 'bar')
    plt.xticks(rotation=90)
    plt.title('Opoziv po kategorijama')
    plt.legend().remove()
    plt.savefig('recall.png')
    
    # FPR
    falsePositiveValuesDF = pd.DataFrame(statisticalData.get("falsePositiveRate"), index = index)
    falsePositiveValuesDF.plot(kind = 'bar')
    plt.xticks(rotation=90)
    plt.title('Stopa pogrešnih klasifikacija po subjektima')
    plt.legend().remove()
    plt.savefig('fpr.png')
    
    # TNR
    trueNegativeValuesDF = pd.DataFrame(statisticalData.get("trueNegativeRate"), index = index)
    trueNegativeValuesDF.plot(kind = 'bar')
    plt.xticks(rotation=90)
    plt.title('Stopa točnih klasifikacija normalnih zapisa po subjektima')
    plt.legend().remove()
    plt.savefig('tnr.png')

    # F-mjera
    fMeassureValuesDF = pd.DataFrame(statisticalData.get("fMeassure"), index = index)
    fMeassureValuesDF.plot(kind = 'bar')
    plt.xticks(rotation=90)
    plt.title('F-mjera modela po subjektima')
    plt.legend().remove()
    plt.savefig('fMeassure.png')

    # Konfuzijska matrica
    disp = ConfusionMatrixDisplay(confusionMatrices, display_labels = model.classes_)
    fig, ax = plt.subplots(figsize = (20,20))
    disp.plot(ax = ax)
    plt.title('Konfuzijska matrica klasifikacija')
    plt.ylabel('Stvarne oznake')
    plt.xlabel('Predviđene oznake')
    plt.xticks(rotation = 90)
    plt.savefig('konfuzijskaMatrica.png')
    return

def saveToExcel(statisticalData: dict, trainingTime, testingTime, index):
    
    # Priprema statistickih podataka za ispis u Excel datoteku
    simpleStatisticsDataFrame = pd.DataFrame(index = ['Vrijeme treniranja', 'Vrijeme testiranja', 'Preciznost', 'Tocnost'])
    simpleStatisticsDataFrame["Jednostavni pokazatelji"] = [trainingTime, testingTime, statisticalData.get("accuracy"), statisticalData.get("precision")]
    classificationStatisticsDataFrame = pd.DataFrame(index = index)
    classificationStatisticsDataFrame['True Positive'] = statisticalData.get("truePositive")
    classificationStatisticsDataFrame['False Positive'] = statisticalData.get("falsePositive")
    classificationStatisticsDataFrame['False Negative'] = statisticalData.get("falseNegative")
    classificationStatisticsDataFrame['True Negative'] = statisticalData.get("trueNegative")
    classificationStatisticsDataFrame['False Positive Rate'] = statisticalData.get("falsePositiveRate")
    classificationStatisticsDataFrame['True Negative Rate'] = statisticalData.get("trueNegativeRate")
    classificationStatisticsDataFrame['Opoziv'] = statisticalData.get("recall")
    classificationStatisticsDataFrame['F-mjera'] = statisticalData.get("fMeassure")
    
    # Kreiranje i ispis statistickih pokazatelja u Excel datoteku
    fileName = 'Statisticki podaci modela.xlsx'
    writer = pd.ExcelWriter(fileName, engine = 'xlsxwriter', engine_kwargs={'options':{'strings_to_formulas': False}})
    workbook = writer.book
    worksheet = workbook.add_worksheet("Statistika")
    worksheet.set_column(0, 0, 20)
    worksheet.set_column(1, 1, 18)
    worksheet.set_column(3, 7, 10)
    worksheet.set_column(8, 11, 12)
    writer.sheets["Statistika"] = worksheet
    simpleStatisticsDataFrame.to_excel(writer, sheet_name = "Statistika", startrow = 0, startcol = 0)
    classificationStatisticsDataFrame.to_excel(writer, sheet_name = "Statistika", startrow = 0, startcol = 3)
    writer.close()
    return