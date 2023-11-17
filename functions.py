from numpy import ravel
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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

def useModel(trainDF_Y, trainDF_X, testDF_Y, testDF_X, model: RandomForestClassifier):
    model.fit(trainDF_X, trainDF_Y.values.ravel())
    prediction = model.predict(testDF_X)
    confusionMatrix = confusion_matrix(testDF_Y, prediction)
    return confusionMatrix