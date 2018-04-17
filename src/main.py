# Load libraries
import os

import numpy as np
import quandl
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

quandl.ApiConfig.api_key = os.environ["API_CREDENTIALS"]


def print_predict_result(y, y_pred):
    p0t0 = 0
    p0t1 = 0
    p1t0 = 0
    p1t1 = 0
    for i in range(len(y)):
        if y_pred[i] == 0 and y[i] == 0:
            p0t0 = p0t0 + 1
        if y_pred[i] == 0 and y[i] == 1:
            p0t1 = p0t1 + 1
        if y_pred[i] == 1 and y[i] == 0:
            p1t0 = p1t0 + 1
        if y_pred[i] == 1 and y[i] == 1:
            p1t1 = p1t1 + 1

    print "  p0t0: " + str(p0t0)
    print "  p0t1: " + str(p0t1)
    if p0t0 + p1t1 > 0:
        print "  p0t0/(p0t0+p0t1): " + str(1.0 * p0t0 / (p0t0 + p0t1))
    else:
        print "  p0t0/(p0t0+p1t1): INF"
    print "  p1t0: " + str(p1t0)
    print "  p1t1: " + str(p1t1)
    if p1t1 + p1t0 > 0:
        print "  p1t1/(p1t1+p1t0): " + str(1.0 * p1t1 / (p1t1 + p1t0) * 100) + '%'
    else:
        print "  p1t1/(p1t1+p1t0): INF"
    if p0t1 + p1t1 > 0:
        print "  p1t1/(p1t1+p0t1): " + str(1.0 * p1t1 / (p1t1 + p0t1) * 100) + '%'
    else:
        print "  p1t1/(p1t1+p1t0): INF"


def print_probabilities(predict_proba):
    probs = {}
    for p in predict_proba:
        proc = str(round(p[1] * 10) / 10.0)
        if probs.has_key(proc):
            probs[proc] = probs[proc] + 1
        else:
            probs[proc] = 1
    print "probabilities: "
    for k in sorted(probs.keys()):
        print "  " + k + ": " + str(probs[k])


# load data
# data = quandl.get('NSE/OIL')
# print("dataset", data.shape)

# GET AMAZON STOCK INF
df = quandl.get("WIKI/AMZN")
# print(df.tail())

# pick closing prices
df = df.reset_index()

# change datecolumn to good or bad 1/0
df['future'] = 0

df['invest'] = 0


def getOneYearValue(df, x):
    todayDate = df['Date'].values[x]
    todayPrice = df['Adj. Close'].values[x]
    futureDate = todayDate + np.timedelta64(365, 'D')
    futureFrame = df[df['Date'].values == futureDate]
    if futureFrame.empty:
        futurePrice = float(0)
    else:
        futurePrice = futureFrame['Adj. Close'].values[0]

    # print( todayPrice)
    # print( futurePrice)

    return futurePrice


for i in range(len(df)):
    df['future'].values[i] = getOneYearValue(df, i)

df = df[df.future != 0]


def isUpTen(df, i):
    toddayprice = df['Adj. Close'].values[i]
    futurePrice = df['future'].values[i]
    if toddayprice * 1.1 < futurePrice:
        return 1
    else:
        return 0


for i in range(len(df)):
    df['invest'].values[i] = isUpTen(df, i)

X = df[['Adj. Close']]
y = df['invest']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(
    n_estimators=100)

# X_forecast = X_forecast.astype(int)

# Train model
clf = clf.fit(X_train, y_train)

# Test model
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print 'STEG 6 # Random Forest'

print "\ny count: " + str(len(y))
print "y_train count: " + str(len(y_train))
print "y_test count: " + str(len(y_test))

print "\ntest\n----------"
print "accuracy score: " + str(accuracy_score(y_test, y_test_pred))
print "roc score: " + str(roc_auc_score(y_test, y_test_pred))
print "\n\n----------"

print "details"
print_predict_result(y_test.values, y_test_pred)
print_probabilities(clf.predict_proba(X_test.values))

print "\n\n----------"

print "\ntraining\n----------"
print "accuracy score: " + str(accuracy_score(y_train, y_train_pred))
print "roc score: " + str(roc_auc_score(y_train, y_train_pred))
print "details"
print_predict_result(y_train.values, y_train_pred)
print_probabilities(clf.predict_proba(X_train.values))
