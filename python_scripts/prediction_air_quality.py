from pickle import FALSE
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics as metrics
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
import json

predict_bicycles = True



df = pd.read_csv('../processed_data/full_data_imputed_daily.csv')


def save(string, n):
    doc = "results_air_quality.txt"
    if(predict_bicycles):
        doc = "results_bicycles.txt"

    with open(doc, "a+") as f:
        f.write(string)
        for i in range(n):
            f.write('\n')
        f.close()


save("MODEL EVALUATION AIR QUALITY", 2)

models = {
    "LinearModel": {"function": linear_model.LinearRegression(),
                    "params": {'n_features_to_select': list(range(2, 11))}
                    },
    "DecisionTree": {
        "function": DecisionTreeRegressor(),
        "params": {"estimator__max_depth": [2,5,None], 
                    'n_features_to_select': list(range(2, 11))}
    },

    "RidgeRegression": {"function": linear_model.Ridge(),
                        "params": {'estimator__alpha': [1, 5, 10, 20, 30, 40, 50, 100, 150, 200, 5, 0.5, 0.1, 0.05, 0.01], 'n_features_to_select': list(range(2, 11))}
                        },
}

# split dataset into a train and test set
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.7
train_data = df[df['is_train']]
test_data = df[~df['is_train']]


X_labels = ['Jahr', 'Monat', 'Tag', 'Zweirad', 'Personenwagen',
           'Lastwagen', 'Hr', 'RainDur', 'T', 'WVs', 'StrGlo', 'p']
y_labels = ['NO2', 'NO', 'NOx', 'O3', 'CO', 'PM10', 'SO2']

if(predict_bicycles):
    X_labels = ['Jahr', 'Monat', 'Tag', 'Personenwagen',
                'Lastwagen', 'Hr', 'RainDur', 'T', 'WVs', 'StrGlo', 'p', 'NO2', 'NO',
        'NOx', 'O3', 'CO', 'PM10', 'SO2']
    y_labels = ['Zweirad']


X_train, X_test = train_data[X_labels], test_data[X_labels]

for label in y_labels:
    save(label, 1)
    save("-------------------------", 2)
    y = train_data[label].sort_index().to_numpy()

    for name, model in models.items():
        save(name, 1)
        regressor = GridSearchCV(
            RFE(model["function"]), model["params"], cv=5, scoring='r2')
        regressor.fit(X_train, y)
        best_score = regressor.best_score_
        best_params = regressor.best_params_
        features = regressor.best_estimator_.ranking_
        save("Best score:"+str(best_score), 1)
        save(json.dumps(best_params), 1)
        save("Features:"+str(features), 2)
