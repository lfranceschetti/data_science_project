import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('../processed_data/full_data_imputed_with_EAQI.csv')
df = df.dropna()
def save(string, n):
    doc = "results_classification_eqi.txt"

    with open(doc, "a+") as f:
        f.write(string)
        for i in range(n):
            f.write('\n')
        f.close()


# --- split dataset into a train and test set ---

X = df[['T','StrGlo','Jahr', 'Monat', 'Tag', 'Zweirad', 'Personenwagen','Lastwagen', 'Hr', 'RainDur',  'WVs', 'p']]  # leave out NO2, PM10_calc,NO2_AQI,PM10_AQI, PM10 and AQI
y = df[["AQI"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100) #RANDOM STATE HERAUSNEHMEN AM SCHLUSS

### --- Linear models: ---
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# - Support Vector Machines (SVM) - -> too computing heavy...

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

### --- Non - Linear models: ---
# - Decision Tree -
from sklearn import tree


params = {
'criterion': ['gini', 'entropy'],
'max_depth' : range(2,5,1),
'min_samples_leaf' : range(1,10,1),
#'min_samples_split': range(2,10,1),
#'splitter' : ['best', 'random']
}
scoring = ['precision_macro', 'recall_macro']

clf = GridSearchCV(
    estimator=tree.DecisionTreeClassifier(),
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='accuracy'
)

clf.fit(X_train[0:50000], y_train[0:50000])
tree_model = clf.best_estimator_
print(clf.best_score_, clf.best_params_)
tree.plot_tree(tree_model,filled=True, fontsize= 4,proportion= True, feature_names= ['T','StrGlo','Jahr', 'Monat', 'Tag', 'Zweirad', 'Personenwagen','Lastwagen', 'Hr', 'RainDur',  'WVs', 'p'],)
plt.show()