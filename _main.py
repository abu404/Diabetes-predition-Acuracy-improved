# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>



import pandas as pd

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
rfc =  RandomForestClassifier(n_estimators=2)
rfc.fit(X,Y)
rfc.predict(np.array([1,1,1,1]))
rfc.predict(X)
main = pd.read_csv("diabetes.csv")
main.head()
col =  main.columns.tolist()
col =  col[0:8]
X = main[col]
X.tail(1)
X.describe()
train =  X[['Glucose','DiabetesPedigreeFunction','Age']]
X.head(1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
Y = main.Outcome
Y[0]
X_train,X_test,y_train,y_test=tts(X,Y,test_size=0.2, random_state=122)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
clf  =  RandomForestClassifier(n_estimators=1000,n_jobs=-1)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score,roc_auc_score
accuracy_score(y_test,pred)
clf.feature_importances_
roc_auc_score(y_test,pred)
for i in range(500,1000):
    clf  =  RandomForestClassifier(n_estimators=i,n_jobs=2,min_samples_leaf=50)
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    print accuracy_score(y_test,pred), roc_auc_score(y_test,pred)
    print clf.feature_importances_
    print "-----------------------------------------------------------------------------"

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint

clf = RandomForestClassifier(n_estimators=1000,n_jobs=-1)

param_dist = {"max_depth": [3, None],
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 40),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
start = time()
random_search.fit(X, Y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

from time import time

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



report(random_search.cv_results_)




