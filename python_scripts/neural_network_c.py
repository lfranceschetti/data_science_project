# --- installing all packages: ---
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from sklearn import metrics


#!
n = 2500
data = pd.read_csv('../processed_data/full_data_imputed_with_EAQI.csv')
data = data[0:n]
for i in data.select_dtypes('object').columns:
    le = LabelEncoder().fit(data[i])
    data[i] = le.transform(data[i])

X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X_data = X_scaler.fit_transform(
    data[['Datum', 'Zweirad', 'Personenwagen', 'Lastwagen', 'Hr', 'RainDur', 'T', 'WVs', 'StrGlo', 'p', 'NO2', 'NO',
          'NOx', 'O3', 'PM10', 'SO2', 'CO']])
Y_data = Y_scaler.fit_transform(data[['CO']])


def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon
    for i in range(start, end):
        indices = range(i - window, i)
        X.append(dataset[indices])
        indicey = range(i + 1, i + 1 + horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y)


hist_window = 200 # TODO wechseln wie unten (1)
horizon = 800
TRAIN_SPLIT = 500
x_train, y_train = custom_ts_multi_data_prep(X_data, Y_data, 0, TRAIN_SPLIT, hist_window, horizon)
x_vali, y_vali = custom_ts_multi_data_prep(X_data, Y_data, TRAIN_SPLIT, None, hist_window, horizon)


batch_size = 256
buffer_size = 150

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
val_data = val_data.batch(batch_size).repeat()

validate = data[['Datum', 'Zweirad', 'Personenwagen', 'Lastwagen', 'Hr', 'RainDur', 'T', 'WVs', 'StrGlo', 'p', 'NO2', 'NO',
          'NOx', 'O3', 'PM10', 'SO2', 'CO']].tail(200) # TODO wechseln wie oben (1)
data.drop(data.tail(10).index,inplace=True)

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True),
                                  input_shape=x_train.shape[-2:]),
    tf.keras.layers.Dense(20, activation='tanh'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
    tf.keras.layers.Dense(20, activation='tanh'),
    tf.keras.layers.Dense(20, activation='tanh'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(units=horizon),
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()

model_path = 'Bidirectional_LSTM_Multivariate.h5'

early_stopings = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min',
                                                verbose=0)
callbacks = [early_stopings, checkpoint]

history = lstm_model.fit(train_data, epochs=5, steps_per_epoch=5, validation_data=val_data, validation_steps=50,
                         verbose=1, callbacks=callbacks)

plt.figure(figsize=(16, 9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'])
plt.show()

data_val = X_scaler.fit_transform(
    data[['Datum', 'Zweirad', 'Personenwagen', 'Lastwagen', 'Hr', 'RainDur', 'T', 'WVs', 'StrGlo', 'p', 'NO2', 'NO',
          'NOx', 'O3', 'PM10', 'SO2','CO' ]].tail(200))  # TODO wechseln wie oben (1)


val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])
pred = lstm_model.predict(val_rescaled)
pred_Inverse = Y_scaler.inverse_transform(pred)
pred_Inverse


def timeseries_evaluation_metrics_func(y_true, y_pred):
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print('Evaluation metric results:-')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}', end='\n\n')


timeseries_evaluation_metrics_func(validate['CO'], pred_Inverse[0])  #TODO ich glaube das sollte jetzt gehen

plt.figure(figsize=(16, 9))
plt.plot(list(pred_Inverse[0]))
plt.plot( list(validate['CO']))
plt.title("Actual vs Predicted")
plt.ylabel("CO")
plt.legend(('Actual', 'predicted'))
plt.show()