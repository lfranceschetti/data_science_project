###############################################
###                Imports                  ###
###############################################
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('../processed_data/full_data_imputed_with_EAQI.csv')
df = df.dropna()

#################################################
### General Settings: (Activate / Deactivate) ###
#################################################

# LINEAR MODELS
#   - 1. Support Vector Machines (SVM)
support_vector_machines = False

# NON-LINEAR MODELS
#   - 1. Decision Tree
decision_tree = False
#   - 2. Knn with CV and Bagging
knn_cv= False


###############################################
###         Save string to textfile         ###
###############################################

def save(string, n):
    doc = "results_classification_eqi.txt"

    with open(doc, "a+") as f:
        f.write(string)
        for i in range(n):
            f.write('\n')
        f.close()

###############################################
#  split dataset into a train and test set  ###
###############################################

X = df[['T','StrGlo','Jahr', 'Monat', 'Tag', 'Zweirad', 'Personenwagen','Lastwagen', 'Hr', 'RainDur',  'WVs', 'p']]  # leave out NO2, PM10_calc,NO2_AQI,PM10_AQI, PM10 and AQI
y = df[["AQI"]]

X_train, X_test, y_train, y_test = train_test_split(X, y)
y_train=y_train.values.ravel()

###############################################
###            LINEAR MODELS                ###
###############################################

###############################################
###     1. Support Vector Machines (SVM)    ###
###############################################
if support_vector_machines == True:
    from sklearn.metrics import accuracy_score
    params = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf']
    }

    clf = GridSearchCV(
        estimator=SVC(),
        param_grid=params,
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    clf.fit(X_train[1:100], y_train[1:100])
    print(clf.best_score_, clf.best_params_)

###############################################
###          NON LINEAR MODELS              ###
###############################################

###############################################
###          1. Decision Tree               ###
###############################################
if decision_tree == True:

    params = {
    'criterion': ['gini', 'entropy'],
    'max_depth' : range(2,10,1),
    #'min_samples_leaf' : range(1,10,1),
    #'min_samples_split': range(2,10,1),
    'splitter' : ['best', 'random']
    }
    clf = GridSearchCV(
        estimator=tree.DecisionTreeClassifier(),
        param_grid=params,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )

    clf.fit(X_train, y_train)
    tree_model = clf.best_estimator_
    print(clf.best_score_, clf.best_params_)
    tree.plot_tree(tree_model,
                   filled=True,
                   fontsize= 4,
                   proportion= True,
                   feature_names= ['T','StrGlo','Jahr', 'Monat', 'Tag', 'Zweirad', 'Personenwagen','Lastwagen', 'Hr', 'RainDur',  'WVs', 'p']
                   )

    print(tree.export_text(tree_model))


###############################################
###      2. Knn with CV and Bagging         ###
###############################################
if knn_cv == True:


    knn = KNeighborsClassifier()
    params = [{'n_neighbors': [40,90,100],
             'weights': ['uniform', 'distance'],
             'leaf_size': [15, 20]}]

    cv_knn = GridSearchCV(knn,
                          param_grid=params,
                          scoring='accuracy',
                          cv=5)

    cv_knn.fit(X_train, y_train)
    print(cv_knn.best_score_, cv_knn.best_params_)

