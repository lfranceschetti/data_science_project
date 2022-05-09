import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics as metrics

df_weather_traffic_daily = pd.read_csv('../processed_data/full_data_imputed_daily.csv')

# correlation_plot = sns.heatmap(df_weather_traffic_daily[['Zweirad','Personenwagen','Lastwagen','Hr',
# 'RainDur','T']].corr(),annot = True,linewidths=3 )
# plt.title("Correlation plot of the dataframe")
# plt.show()


# split dataset into a train and test set
df_weather_traffic_daily['is_train'] = np.random.uniform(0, 1, len(df_weather_traffic_daily)) <= 0.75
train_data = df_weather_traffic_daily[df_weather_traffic_daily['is_train']]
test_data = df_weather_traffic_daily[~df_weather_traffic_daily['is_train']]

print('-----------------------------------------------\nLength of train data:', len(train_data))
print('Length of test data:', len(test_data))

y_train = train_data[['Zweirad']]
x_train = train_data[
    ['Jahr', 'Monat', 'Tag', 'Personenwagen', 'Lastwagen', 'Hr', 'RainDur', 'T', 'WVs', 'StrGlo', 'p', 'NO2', 'NO',
     'NOx', 'O3', 'CO', 'PM10', 'SO2']]

y_test = test_data[['Zweirad']]
x_test = test_data[
    ['Jahr', 'Monat', 'Tag', 'Personenwagen', 'Lastwagen', 'Hr', 'RainDur', 'T', 'WVs', 'StrGlo', 'p', 'NO2', 'NO',
     'NOx', 'O3', 'CO', 'PM10', 'SO2']]

dt = DecisionTreeRegressor()

dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)


# evaluate the model
def regression_results(name_of_model='MODEL EVALUATION', what_predict='PREDICTION', y_true=0, y_pred=0):
    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print(f'\n-------- {name_of_model} -------')
    print(f'--- {what_predict} ---')
    print('|>>> explained_variance: ', round(explained_variance, 4))
    print('|>>> r2:   ', round(r2, 4))
    print('|>>> MAE:  ', round(mean_absolute_error, 4))
    print('|>>> MSE:  ', round(mse, 4), )
    print('|>>> RMSE: ', round(np.sqrt(mse), 4))
    print(f'------{len(name_of_model) * "-"}-----------')


regression_results('RF - Regressor', 'Bicycle and Motorcycle', y_true=y_test, y_pred=y_pred_dt)

### linear model (without cross validation) ###
from sklearn import linear_model

lm = linear_model.LinearRegression()
lm.fit(x_train, y_train)

y_pred_dt = lm.predict(x_test)
regression_results('Linear Model (no CV)', 'Bicycle and Motorcycle', y_true=y_test, y_pred=y_pred_dt)

### linear model (with cross validation) ###
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV

# creating a KFold object with 5 splits #
folds = KFold(n_splits=5, shuffle=True)
hyper_params = [{'n_features_to_select': list(range(2, 11))}]
lm = linear_model.LinearRegression()
lm.fit(x_train, y_train)

rfe = RFE(lm)

# GridSearchCV()
model_cv = GridSearchCV(estimator=rfe,
                        param_grid=hyper_params,
                        scoring='r2',
                        cv=folds,
                        verbose=1,
                        return_train_score=True)

model_cv.fit(x_train, y_train)

cv_results = pd.DataFrame(model_cv.cv_results_)

# plot cv_results
plt.figure(figsize=(16, 6))
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')
plt.show()

# final model with optimal feature counts
lm = linear_model.LinearRegression()
lm.fit(x_train, y_train)

rfe = RFE(lm, n_features_to_select=10)
rfe = rfe.fit(x_train, y_train)

# prediction
y_pred_dt = model_cv.predict(x_test)
regression_results('Linear Model (with CV)', 'Bicycle and Motorcycle', y_true=y_test, y_pred=y_pred_dt)

### JETZT NOCH MIT KATEGORISIERUNG (Linear Model) ###

# df_weather_traffic_daily[['Zweirad']].describe()
# output:
#            Zweirad
# 25%     587.500000
# 50%     881.000000
# 75%    1202.500000

df_weather_traffic_daily.loc[df_weather_traffic_daily['Zweirad'] < 587.5, 'Zweirad'] = 1
df_weather_traffic_daily.loc[
    (df_weather_traffic_daily['Zweirad'] < 881.000000) & (df_weather_traffic_daily['Zweirad'] >= 587.5), 'Zweirad'] = 2
df_weather_traffic_daily.loc[
    (df_weather_traffic_daily['Zweirad'] < 1202.500000) & (df_weather_traffic_daily['Zweirad'] >= 881), 'Zweirad'] = 3
df_weather_traffic_daily.loc[df_weather_traffic_daily['Zweirad'] >= 1202.500000, 'Zweirad'] = 4

# Split data again:
df_weather_traffic_daily['is_train'] = np.random.uniform(0, 1, len(df_weather_traffic_daily)) <= 0.75
train_data = df_weather_traffic_daily[df_weather_traffic_daily['is_train']]
test_data = df_weather_traffic_daily[~df_weather_traffic_daily['is_train']]

y_train = train_data[['Zweirad']]
x_train = train_data[
    ['Jahr', 'Monat', 'Tag', 'Personenwagen', 'Lastwagen', 'Hr', 'RainDur', 'T', 'WVs', 'StrGlo', 'p', 'NO2', 'NO',
     'NOx', 'O3', 'CO', 'PM10', 'SO2']]

y_test = test_data[['Zweirad']]
x_test = test_data[
    ['Jahr', 'Monat', 'Tag', 'Personenwagen', 'Lastwagen', 'Hr', 'RainDur', 'T', 'WVs', 'StrGlo', 'p', 'NO2', 'NO',
     'NOx', 'O3', 'CO', 'PM10', 'SO2']]
lm = linear_model.LinearRegression()
lm.fit(x_train, y_train)

y_pred_dt = lm.predict(x_test)
regression_results('Linear Model (no CV)', 'Bicycle and Motorcycle in categories', y_true=y_test, y_pred=y_pred_dt)