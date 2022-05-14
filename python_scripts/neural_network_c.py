# --- installing all packages: ---
from cProfile import label
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

###################################
#        Model configuaration     #
###################################

data = pd.read_csv('../processed_data/full_data_imputed.csv')

hist_window = 48
horizon = 48
TRAIN_SPLIT = int(0.7 * data.shape[0])
batch_size = 5
buffer_size = 150
num_epochs = 5
steps_per_epoch = None

gases = ['NO2', 'NO','NOx', 'O3', 'PM10', 'SO2', 'CO']
x_labels = ['Zweirad', 'Personenwagen', 'Lastwagen', 'Hr', 'RainDur', 'T', 'WVs', 'StrGlo', 'p'] 
gas = "O3"

###################################
#        Data preprocessing       #
###################################

x_labels.append(gas)

for i in data.select_dtypes('object').columns:
    le = LabelEncoder().fit(data[i])
    data[i] = le.transform(data[i])

X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X_data = X_scaler.fit_transform(data[x_labels])
Y_data = Y_scaler.fit_transform(data[[gas]])


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

x_train, y_train = custom_ts_multi_data_prep(X_data, Y_data, 0, TRAIN_SPLIT, hist_window, horizon)
x_vali, y_vali = custom_ts_multi_data_prep(X_data, Y_data, TRAIN_SPLIT, None, hist_window, horizon)

print(x_train.shape)
print(y_train.shape)
print(x_vali.shape)
print(y_vali.shape)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(buffer_size).batch(batch_size)
val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
val_data = val_data.batch(batch_size)
validate = data[x_labels].tail(hist_window)
data.drop(data.tail(horizon).index,inplace=True)



###################################
#              Model              #
###################################


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

history = lstm_model.fit(train_data, epochs=num_epochs, steps_per_epoch=steps_per_epoch, validation_data=val_data, validation_steps=50,
                        verbose=1, callbacks=callbacks)

plt.figure(figsize=(16, 9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'])
plt.show()

print(history)
#lstm_model = keras.models.load_model('Bidirectional_LSTM_Multivariate.h5')

#pred = lstm_model.predict(train_data)


data_val = X_scaler.fit_transform(data[x_labels].tail(horizon))  


val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])
pred = lstm_model.predict(val_rescaled)
pred_Inverse = Y_scaler.inverse_transform(pred)



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


timeseries_evaluation_metrics_func(validate[gas], pred_Inverse[0]) 

plt.figure(figsize=(16, 9))
plt.plot(list(pred_Inverse[0]), color="red")
plt.plot(list(validate[gas]), color="black")
plt.title("Actual vs Predicted")
plt.ylabel("CO")
plt.legend(('predicted','Actual'))
plt.show()