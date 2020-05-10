import datetime
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
# from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, BatchNormalization
# from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import TensorBoard
from keras import callbacks
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.linear_model import LinearRegression
import keras

# Fix random seed
np.random.seed(5)

def create_dataset_pd(data, lag=4):
    dataX, dataY = [], []

    for i in range(len(data) - lag):
        dataX.append(data.iloc[i:(i+lag)])
        dataY.append(data.iloc[i+lag])

    return np.array(dataX), np.array(dataY)


def create_dataset_np(data, lag=4):
    dataX, dataY = [], []

    for i in range(len(data) - lag):
        dataX.append(data[i:(i+lag)])
        dataY.append(data[i+lag])

    return np.array(dataX), np.array(dataY)


class CustomHistory(callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))


def stack_lstm():
    data_path = r'C:\Users\CJ\workspace\Computational_finance'
    data_path = os.path.join(data_path, 'total.xlsx')
    data = pd.read_excel(data_path)
    # kospi_last = data['KOSPI_PX_LAST']
    # kospi_open = data['KOSPI_PX_OPEN']
    kospi_low = data['KOSPI_PX_LOW']
    kospi_high = data['KOSPI_PX_HIGH']
    data['Volatility'] = kospi_high - kospi_low
    df = data['Volatility'].dropna()

    # Hyper-parameter
    loof_back = 8
    iteration = 100

    # Data preprocessing
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.values
    df = df.reshape(-1, 1)
    df = scaler.fit_transform(df)
    '''
    df = df.values
    df = df.reshape(-1, 1)
    df = (df - np.mean(df)) / np.std(df)

    train = df[:int(len(df)*0.8)]
    val = df[int(len(df)*0.8):int(len(df)*0.9)]
    test = df[int(len(df)*0.9):]

    x_train, y_train = create_dataset_np(train, loof_back)
    x_val, y_val = create_dataset_np(val, loof_back)
    x_test, y_test = create_dataset_np(test, loof_back)

    # Feature data preprocessing
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Model designing
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(1, loof_back, 1), stateful=True, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32, batch_input_shape=(1, loof_back, 1), stateful=True))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    # Set Learning config
    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    # Create TensorBoard callback
    tb_hist = TensorBoard(log_dir='./graph/stack_lstm_normsc_lag{}_iter{}'.format(loof_back, iteration), histogram_freq=0, write_grads=True, write_images=True)

    # Learning model
    custom_hist = CustomHistory()
    custom_hist.init()
    for i in range(iteration):
        print("--------------------------")
        print("------------{}------------".format(i))
        model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False,
                  validation_data=(x_val, y_val), callbacks=[tb_hist, custom_hist])

    plt.plot(custom_hist.train_loss, 'b', label='train loss')
    plt.plot(custom_hist.val_loss, 'r', label='val loss')

    model.save('./model/stack_lstm_normsc_lag{}_iter{}.h5'.format(loof_back, iteration))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # Model Evaluation
    trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
    model.reset_states()
    print('Train Score: ', trainScore)
    valScore = model.evaluate(x_val, y_val, batch_size=1, verbose=0)
    model.reset_states()
    print('Validation Score: ', valScore)
    testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
    model.reset_states()
    print('Test Score: ', testScore)

    plt.savefig('./graph/stack_normsc_lag{}_iter{}'.format(loof_back, iteration))


def stack_lstm_2atr():
    data_path = r'C:\Users\CJ\workspace\Computational_finance'
    data_path = os.path.join(data_path, 'total_merged.xlsx')
    data = pd.read_excel(data_path)
    kospi_last = data['KOSPI_PX_LAST']
    kospi_open = data['KOSPI_PX_OPEN']
    kospi_low = data['KOSPI_PX_LOW']
    kospi_high = data['KOSPI_PX_HIGH']
    data['Volatility'] = kospi_high - kospi_low
    data['Trend'] = (kospi_last - kospi_open)/(kospi_high - kospi_low)
    df_v = data['Volatility'].dropna()
    df_t = data['Trend'].dropna()
    df = np.array([df_v, df_t])
    df = df.transpose()

    loof_back = 5
    iteration = 25

    # Data preprocessing
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.values
    df = df.reshape(-1, 1)
    df = scaler.fit_transform(df)
    '''
    df = (df - np.mean(df)) / np.std(df)

    train = df[:int(len(df)*0.8)]
    val = df[int(len(df)*0.8):int(len(df)*0.9)]
    test = df[int(len(df)*0.9):]

    x_train, y_train = create_dataset_np(train, loof_back)
    x_val, y_val = create_dataset_np(val, loof_back)
    x_test, y_test = create_dataset_np(test, loof_back)

    # Feature data preprocessing
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Model designing
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(1, loof_back, 2), stateful=True, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32, batch_input_shape=(1, loof_back, 2), stateful=True))
    model.add(Dropout(0.3))
    model.add(Dense(2))

    # Set Learning config
    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    # Create TensorBoard callback
    tb_hist = TensorBoard(log_dir='./graph/stack_lstm_normsc_2atr_lag{}_iter{}'.format(loof_back, iteration), histogram_freq=0, write_grads=True, write_images=True)

    # Learning model
    custom_hist = CustomHistory()
    custom_hist.init()
    for i in range(iteration):
        print("--------------------------")
        print("------------{}------------".format(i))
        model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False,
                  validation_data=(x_val, y_val), callbacks=[tb_hist, custom_hist])

    plt.plot(custom_hist.train_loss, 'b', label='train loss')
    plt.plot(custom_hist.val_loss, 'r', label='val loss')

    model.save('./model/stack_lstm_normsc_2atr_lag{}_iter{}.h5'.format(loof_back, iteration))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # Model Evaluation
    trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
    model.reset_states()
    print('Train Score: ', trainScore)
    valScore = model.evaluate(x_val, y_val, batch_size=1, verbose=0)
    model.reset_states()
    print('Validation Score: ', valScore)
    testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
    model.reset_states()
    print('Test Score: ', testScore)

    plt.savefig('./graph/stack_normsc_2atr_lag{}_iter{}'.format(loof_back, iteration))


def stack_lstm_2atr_mv():
    data_path = r'C:\Users\CJ\workspace\Computational_finance'
    # data_path = os.path.join(data_path, 'Data_set.csv')
    data_path = os.path.join(data_path, 'total_merged.xlsx')
    data = pd.read_excel(data_path)

    endo_key = ['KOSPI']
    # exo_key = ['GOLD_PRICE', 'HSI', 'JPYKRW', 'NASDAQ', 'OIL', 'RTY', 'SHI', 'SOX', 'SPX', 'USDKRW', 'EM_INDEX']
    # exo_key = ['HSI', 'JPYKRW', 'SPX', 'USDKRW', 'EM_INDEX']
    exo_key = ['HSI', 'JPYKRW', 'SPX', 'DOLLAR_INDEX', 'EM_INDEX', 'GOLD_PRICE', 'RTY', 'OIL', 'USDKRW']
    attr_key = ['PX_HIGH', 'PX_LOW', 'PX_OPEN', 'PX_LAST']
    tot_key = []

    for a in (exo_key + endo_key):
        for b in attr_key:
            tot_key.append(a + '_' + b)

    df = data.loc[:, tot_key]
    df = df.dropna()
    for a in (endo_key + exo_key):
        vol_key = a + '_VOL'
        hi_key = a + '_PX_HIGH'
        lo_key = a + '_PX_LOW'
        df[vol_key] = df[hi_key] - df[lo_key]

    exo_list = [df[key] for key in [x + '_VOL' for x in exo_key + endo_key]]

    kospi_last = df['KOSPI_PX_LAST']
    kospi_open = df['KOSPI_PX_OPEN']
    kospi_low = df['KOSPI_PX_LOW']
    kospi_high = df['KOSPI_PX_HIGH']
    df['KOSPI_TREND'] = (kospi_last - kospi_open)/(kospi_high - kospi_low)

    # df_np = np.array([df_v, df_t] + exo_list)
    # df_np = df_np.transpose()

    df_y = np.array([df['KOSPI_VOL'], df['KOSPI_TREND']]).transpose()
    df_x = np.array(exo_list).transpose()

    # df_y = df_y[1:, :]
    # df_x = df_x[:-1, :]

    # loof_back = df_x.shape[1]
    loof_back = 5
    iteration = 15

    # Data preprocessing
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.values
    df = df.reshape(-1, 1)
    df = scaler.fit_transform(df)
    '''

    data_x, data_y = [], []
    for i in range(len(df_x) - loof_back - 1):
        data_x.append(df_x[i:(i + loof_back)])
        data_y.append(df_y[i + loof_back + 1])

    df_x = data_x
    df_y = data_y

    df_y = (df_y - np.mean(df_y, axis=0)) / np.std(df_y, axis=0)
    df_x = (df_x - np.mean(df_x, axis=0)) / np.std(df_x, axis=0)

    ratio_8 = int(len(df_y) * 0.8)
    ratio_9 = int(len(df_y) * 0.9)

    x_train = df_x[:ratio_8]
    x_val = df_x[ratio_8:ratio_9]
    x_test = df_x[ratio_9:]

    y_train = df_y[:ratio_8]
    y_val = df_y[ratio_8:ratio_9]
    y_test = df_y[ratio_9:]

    # Feature data preprocessing
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Model designing
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(1, loof_back, len(exo_key + endo_key)), stateful=True, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32, batch_input_shape=(1, loof_back, len(exo_key + endo_key)), stateful=True))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    # Set Learning config
    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    # Create TensorBoard callback
    tb_hist = TensorBoard(log_dir='./graph/64stack_lstm_normsc_2atr_mv_lag{}_iter{}'.format(loof_back, iteration), histogram_freq=0, write_grads=True, write_images=True)

    # Learning model
    custom_hist = CustomHistory()
    custom_hist.init()
    for i in range(iteration):
        print("--------------------------")
        print("------------{}------------".format(i))
        model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False,
                  validation_data=(x_val, y_val), callbacks=[tb_hist, custom_hist])

    plt.plot(custom_hist.train_loss, 'b', label='train loss')
    plt.plot(custom_hist.val_loss, 'r', label='val loss')

    model.save('./model/stack_lstm_normsc_2atr_mv_lag{}_iter{}.h5'.format(loof_back, iteration))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # Model Evaluation
    trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
    model.reset_states()
    print('Train Score: ', trainScore)
    valScore = model.evaluate(x_val, y_val, batch_size=1, verbose=0)
    model.reset_states()
    print('Validation Score: ', valScore)
    testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
    model.reset_states()
    print('Test Score: ', testScore)

    plt.savefig('./graph/64stack_normsc_2atr_mv_lag{}_iter{}'.format(loof_back, iteration))


def state_lstm():
    data_path = r'C:\Users\CJ\workspace\Computational_finance'
    data_path = os.path.join(data_path, 'total_merged.xlsx')
    data = pd.read_excel(data_path)
    kospi_last = data['KOSPI_PX_LAST']
    kospi_open = data['KOSPI_PX_OPEN']
    kospi_low = data['KOSPI_PX_LOW']
    kospi_high = data['KOSPI_PX_HIGH']
    data['Volatility'] = kospi_high - kospi_low
    df = data['Volatility'].dropna()

    loof_back = 5
    iteration = 50

    # Data preprocessing
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.values
    df = df.reshape(-1, 1)
    df = scaler.fit_transform(df)
    '''
    df = df.values
    df = df.reshape(-1, 1)
    df = (df - np.mean(df)) / np.std(df)

    train = df[:int(len(df)*0.8)]
    val = df[int(len(df)*0.8):int(len(df)*0.9)]
    test = df[int(len(df)*0.9):]

    x_train, y_train = create_dataset_np(train, loof_back)
    x_val, y_val = create_dataset_np(val, loof_back)
    x_test, y_test = create_dataset_np(test, loof_back)

    # Feature data preprocessing
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Model designing
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(1, loof_back, 1), stateful=True))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    # Set Learning config
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Create TensorBoard callback
    tb_hist = TensorBoard(log_dir='./graph/state_lstm_normsc_lag{}_iter{}'.format(loof_back, iteration), histogram_freq=0, write_grads=True, write_images=True)

    # Model Learning
    custom_hist = CustomHistory()
    custom_hist.init()

    for i in range(iteration):
        print("--------------------------")
        print("------------{}------------".format(i))
        model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False,
                  validation_data=(x_val, y_val), callbacks=[tb_hist, custom_hist])

    plt.plot(custom_hist.train_loss, 'b', label='train loss')
    plt.plot(custom_hist.val_loss, 'r', label='val loss')

    model.save('./model/state_lstm_normsc_lag{}_iter{}.h5'.format(loof_back, iteration))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # Model Evaluation
    trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
    model.reset_states()
    print('Train Score: ', trainScore)
    valScore = model.evaluate(x_val, y_val, batch_size=1, verbose=0)
    model.reset_states()
    print('Validation Score: ', valScore)
    testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
    model.reset_states()
    print('Test Score: ', testScore)

    plt.savefig('./graph/state_normsc_lag{}_iter{}'.format(loof_back, iteration))


def state_lstm_2atr():
    data_path = r'C:\Users\CJ\workspace\Computational_finance'
    data_path = os.path.join(data_path, 'total_merged.xlsx')
    data = pd.read_excel(data_path)
    kospi_last = data['KOSPI_PX_LAST']
    kospi_open = data['KOSPI_PX_OPEN']
    kospi_low = data['KOSPI_PX_LOW']
    kospi_high = data['KOSPI_PX_HIGH']
    data['Volatility'] = kospi_high - kospi_low
    data['Trend'] = (kospi_last - kospi_open)/(kospi_high - kospi_low)
    df_v = data['Volatility'].dropna()
    df_t = data['Trend'].dropna()
    df = np.array([df_v, df_t])
    df = df.transpose()

    loof_back = 5
    iteration = 25

    # Data preprocessing
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.values
    df = df.reshape(-1, 1)
    df = scaler.fit_transform(df)
    '''
    df = (df - np.mean(df)) / np.std(df)

    train = df[:int(len(df)*0.8)]
    val = df[int(len(df)*0.8):int(len(df)*0.9)]
    test = df[int(len(df)*0.9):]

    x_train, y_train = create_dataset_np(train, loof_back)
    x_val, y_val = create_dataset_np(val, loof_back)
    x_test, y_test = create_dataset_np(test, loof_back)

    # Feature data preprocessing
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Model designing
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(1, loof_back, 2), stateful=True))
    model.add(Dropout(0.3))
    model.add(Dense(2))

    # Set Learning config
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Create TensorBoard callback
    tb_hist = TensorBoard(log_dir='./graph/32state_lstm_normsc_2atr_lag{}_iter{}'.format(loof_back, iteration), histogram_freq=0, write_grads=True, write_images=True)

    # Model Learning
    custom_hist = CustomHistory()
    custom_hist.init()

    for i in range(iteration):
        print("--------------------------")
        print("------------{}------------".format(i))
        model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False,
                  validation_data=(x_val, y_val), callbacks=[tb_hist, custom_hist])

    plt.plot(custom_hist.train_loss, 'b', label='train loss')
    plt.plot(custom_hist.val_loss, 'r', label='val loss')

    model.save('./model/32state_lstm_normsc_2atr_lag{}_iter{}.h5'.format(loof_back, iteration))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # Model Evaluation
    trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
    model.reset_states()
    print('Train Score: ', trainScore)
    valScore = model.evaluate(x_val, y_val, batch_size=1, verbose=0)
    model.reset_states()
    print('Validation Score: ', valScore)
    testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
    model.reset_states()
    print('Test Score: ', testScore)

    plt.savefig('./graph/32state_normsc_2atr_lag{}_iter{}'.format(loof_back, iteration))


def state_lstm_2atr_mv():
    data_path = r'C:\Users\CJ\workspace\Computational_finance'
    # data_path = os.path.join(data_path, 'Data_set.csv')
    data_path = os.path.join(data_path, 'total_merged.xlsx')
    data = pd.read_excel(data_path)

    endo_key = ['KOSPI']
    # exo_key = ['GOLD_PRICE', 'HSI', 'JPYKRW', 'NASDAQ', 'OIL', 'RTY', 'SHI', 'SOX', 'SPX', 'USDKRW', 'EM_INDEX']
    # exo_key = ['HSI', 'JPYKRW', 'SPX', 'USDKRW', 'EM_INDEX']
    exo_key = ['HSI', 'JPYKRW', 'SPX', 'DOLLAR_INDEX', 'EM_INDEX', 'GOLD_PRICE', 'RTY', 'OIL', 'USDKRW']
    attr_key = ['PX_HIGH', 'PX_LOW', 'PX_OPEN', 'PX_LAST']
    tot_key = []

    for a in (exo_key + endo_key):
        for b in attr_key:
            tot_key.append(a + '_' + b)

    df = data.loc[:, tot_key]
    df = df.dropna()
    for a in (endo_key + exo_key):
        vol_key = a + '_VOL'
        hi_key = a + '_PX_HIGH'
        lo_key = a + '_PX_LOW'
        df[vol_key] = df[hi_key] - df[lo_key]

    exo_list = [df[key] for key in [x + '_VOL' for x in exo_key + endo_key]]

    kospi_last = df['KOSPI_PX_LAST']
    kospi_open = df['KOSPI_PX_OPEN']
    kospi_low = df['KOSPI_PX_LOW']
    kospi_high = df['KOSPI_PX_HIGH']
    df['KOSPI_TREND'] = (kospi_last - kospi_open)/(kospi_high - kospi_low)

    # df_np = np.array([df_v, df_t] + exo_list)
    # df_np = df_np.transpose()

    df_y = np.array([df['KOSPI_VOL'], df['KOSPI_TREND']]).transpose()
    df_x = np.array(exo_list).transpose()

    # df_y = df_y[1:, :]
    # df_x = df_x[:-1, :]

    # loof_back = df_x.shape[1]
    loof_back = 5
    iteration = 15

    # Data preprocessing
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.values
    df = df.reshape(-1, 1)
    df = scaler.fit_transform(df)
    '''

    data_x, data_y = [], []
    for i in range(len(df_x) - loof_back - 1):
        data_x.append(df_x[i:(i + loof_back)])
        data_y.append(df_y[i + loof_back + 1])

    df_x = data_x
    df_y = data_y

    df_y = (df_y - np.mean(df_y, axis=0)) / np.std(df_y, axis=0)
    df_x = (df_x - np.mean(df_x, axis=0)) / np.std(df_x, axis=0)

    ratio_8 = int(len(df_y) * 0.8)
    ratio_9 = int(len(df_y) * 0.9)

    x_train = df_x[:ratio_8]
    x_val = df_x[ratio_8:ratio_9]
    x_test = df_x[ratio_9:]

    y_train = df_y[:ratio_8]
    y_val = df_y[ratio_8:ratio_9]
    y_test = df_y[ratio_9:]

    # Feature data preprocessing
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Model designing
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(1, loof_back, len(exo_key)+len(endo_key)), stateful=True))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    # Set Learning config
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Create TensorBoard callback
    tb_hist = TensorBoard(log_dir='./graph/state_lstm_normsc_2atr_mv_lag{}_iter{}'.format(loof_back, iteration), histogram_freq=0, write_grads=True, write_images=True)

    # Model Learning
    custom_hist = CustomHistory()
    custom_hist.init()

    for i in range(iteration):
        print("--------------------------")
        print("------------{}------------".format(i))
        model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False,
                  validation_data=(x_val, y_val), callbacks=[tb_hist, custom_hist])

    plt.plot(custom_hist.train_loss, 'b', label='train loss')
    plt.plot(custom_hist.val_loss, 'r', label='val loss')

    model.save('./model/state_lstm_normsc_2atr_mv_lag{}_iter{}.h5'.format(loof_back, iteration))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # Model Evaluation
    trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
    model.reset_states()
    print('Train Score: ', trainScore)
    valScore = model.evaluate(x_val, y_val, batch_size=1, verbose=0)
    model.reset_states()
    print('Validation Score: ', valScore)
    testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
    model.reset_states()
    print('Test Score: ', testScore)

    plt.savefig('./graph/state_normsc_2atr_mv_lag{}_iter{}'.format(loof_back, iteration))


def simple_lstm():
    data_path = r'C:\Users\CJ\workspace\Computational_finance'
    data_path = os.path.join(data_path, 'total_merged.xlsx')
    data = pd.read_excel(data_path)
    kospi_last = data['KOSPI_PX_LAST']
    kospi_open = data['KOSPI_PX_OPEN']
    kospi_low = data['KOSPI_PX_LOW']
    kospi_high = data['KOSPI_PX_HIGH']
    data['Volatility'] = kospi_high - kospi_low
    df = data['Volatility'].dropna()

    loof_back = 5
    iteration = 50

    # Data preprocessing
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.values
    df = df.reshape(-1, 1)
    df = scaler.fit_transform(df)
    '''
    df = df.values
    df = df.reshape(-1, 1)
    df_mean = np.mean(df)
    df_std = np.std(df)
    df = (df - np.mean(df)) / np.std(df)

    train = df[:int(len(df)*0.8)]
    val = df[int(len(df)*0.8):int(len(df)*0.9)]
    test = df[int(len(df)*0.9):]

    x_train, y_train = create_dataset_np(train, loof_back)
    x_val, y_val = create_dataset_np(val, loof_back)
    x_test, y_test = create_dataset_np(test, loof_back)

    # Feature data preprocessing
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Model designing
    model = Sequential()
    model.add(LSTM(32, input_shape=(None, 1)))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    # Set Learning config
    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    # Create TensorBoard callback
    tb_hist = TensorBoard(log_dir='./graph/simple_lstm_normalscaling_lag{}_iter{}'.format(loof_back, iteration), histogram_freq=0, write_grads=True, write_images=True)

    hist = model.fit(x_train, y_train, epochs=iteration,  batch_size=10, validation_data=(x_val, y_val), callbacks=[tb_hist])

    plt.plot(hist.history['loss'], 'y', label='train loss')
    plt.plot(hist.history['val_loss'], 'r', label='val loss')

    model.save('./model/simple_lstm_normalscaling_model_lag{}_iter{}.h5'.format(loof_back, iteration))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # Model Evaluation
    trainScore = model.evaluate(x_train, y_train, verbose=0)
    model.reset_states()
    print('Train Score: ', trainScore)
    valScore = model.evaluate(x_val, y_val, verbose=0)
    model.reset_states()
    print('Validation Score: ', valScore)
    testScore = model.evaluate(x_test, y_test, verbose=0)
    model.reset_states()
    print('Test Score: ', testScore)

    plt.savefig('./graph/simple_normsc_lag{}_iter{}'.format(loof_back, iteration))


def simple_lstm_2atr_bn():
    data_path = r'C:\Users\CJ\workspace\Computational_finance'
    data_path = os.path.join(data_path, 'total_merged.xlsx')
    data = pd.read_excel(data_path)
    kospi_last = data['KOSPI_PX_LAST']
    kospi_open = data['KOSPI_PX_OPEN']
    kospi_low = data['KOSPI_PX_LOW']
    kospi_high = data['KOSPI_PX_HIGH']
    data['Volatility'] = kospi_high - kospi_low
    data['Trend'] = (kospi_last - kospi_open)/(kospi_high - kospi_low)
    df_v = data['Volatility'].dropna()
    df_t = data['Trend'].dropna()
    df = np.array([df_v, df_t])
    df = df.transpose()

    loof_back = 5
    iteration = 50

    # Data preprocessing
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.values
    df = df.reshape(-1, 1)
    df = scaler.fit_transform(df)
    '''
    df = (df - np.mean(df)) / np.std(df)

    train = df[:int(len(df)*0.8)]
    val = df[int(len(df)*0.8):int(len(df)*0.9)]
    test = df[int(len(df)*0.9):]

    x_train, y_train = create_dataset_np(train, loof_back)
    x_val, y_val = create_dataset_np(val, loof_back)
    x_test, y_test = create_dataset_np(test, loof_back)

    # Feature data preprocessing
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Model designing
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(1, loof_back, 2)))
    # model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(2))

    # Set Learning config
    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    # Create TensorBoard callback
    tb_hist = TensorBoard(log_dir='./graph/simple_lstm_normalscaling_attr2_lag{}_iter{}_bn'.format(loof_back, iteration), histogram_freq=0, write_grads=True, write_images=True)

    hist = model.fit(x_train, y_train, epochs=iteration,  batch_size=1, validation_data=(x_val, y_val), callbacks=[tb_hist])

    plt.plot(hist.history['loss'], 'y', label='train loss')
    plt.plot(hist.history['val_loss'], 'r', label='val loss')

    model.save('./model/simple_lstm_normalscaling_attr2_lag{}_iter{}_bn.h5'.format(loof_back, iteration))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # Model Evaluation
    trainScore = model.evaluate(x_train, y_train, verbose=0)
    model.reset_states()
    print('Train Score: ', trainScore)
    valScore = model.evaluate(x_val, y_val, verbose=0)
    model.reset_states()
    print('Validation Score: ', valScore)
    testScore = model.evaluate(x_test, y_test, verbose=0)
    model.reset_states()
    print('Test Score: ', testScore)

    plt.savefig('./graph/simple_normsc_lag{}_iter{}'.format(loof_back, iteration))


def simple_lstm_2atr():
    data_path = r'C:\Users\CJ\workspace\Computational_finance'
    data_path = os.path.join(data_path, 'total_merged.xlsx')
    data = pd.read_excel(data_path)
    kospi_last = data['KOSPI_PX_LAST']
    kospi_open = data['KOSPI_PX_OPEN']
    kospi_low = data['KOSPI_PX_LOW']
    kospi_high = data['KOSPI_PX_HIGH']
    data['Volatility'] = kospi_high - kospi_low
    data['Trend'] = (kospi_last - kospi_open)/(kospi_high - kospi_low)
    df_v = data['Volatility'].dropna()
    df_t = data['Trend'].dropna()
    df = np.array([df_v, df_t])
    df = df.transpose()

    loof_back = 5
    iteration = 50

    # Data preprocessing
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.values
    df = df.reshape(-1, 1)
    df = scaler.fit_transform(df)
    '''
    df = (df - np.mean(df)) / np.std(df)

    train = df[:int(len(df)*0.8)]
    val = df[int(len(df)*0.8):int(len(df)*0.9)]
    test = df[int(len(df)*0.9):]

    x_train, y_train = create_dataset_np(train, loof_back)
    x_val, y_val = create_dataset_np(val, loof_back)
    x_test, y_test = create_dataset_np(test, loof_back)

    # Feature data preprocessing
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Model designing
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(1, loof_back, 2)))
    model.add(Dense(2))

    # Set Learning config
    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    # Create TensorBoard callback
    tb_hist = TensorBoard(log_dir='./graph/simple_lstm_normalscaling_attr2_lag{}_iter{}'.format(loof_back, iteration), histogram_freq=0, write_grads=True, write_images=True)

    hist = model.fit(x_train, y_train, epochs=iteration,  batch_size=1, validation_data=(x_val, y_val), callbacks=[tb_hist])

    plt.plot(hist.history['loss'], 'y', label='train loss')
    plt.plot(hist.history['val_loss'], 'r', label='val loss')

    model.save('./model/simple_lstm_normalscaling_attr2_lag{}_iter{}.h5'.format(loof_back, iteration))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # Model Evaluation
    trainScore = model.evaluate(x_train, y_train, verbose=0)
    model.reset_states()
    print('Train Score: ', trainScore)
    valScore = model.evaluate(x_val, y_val, verbose=0)
    model.reset_states()
    print('Validation Score: ', valScore)
    testScore = model.evaluate(x_test, y_test, verbose=0)
    model.reset_states()
    print('Test Score: ', testScore)

    plt.savefig('./graph/simple_normsc_lag{}_iter{}'.format(loof_back, iteration))


def simple_lstm_2atr_mv():
    data_path = r'C:\Users\CJ\workspace\Computational_finance'
    # data_path = os.path.join(data_path, 'Data_set.csv')
    data_path = os.path.join(data_path, 'total_merged.xlsx')
    data = pd.read_excel(data_path)

    endo_key = ['KOSPI']
    # exo_key = ['GOLD_PRICE', 'HSI', 'JPYKRW', 'NASDAQ', 'OIL', 'RTY', 'SHI', 'SOX', 'SPX', 'USDKRW', 'EM_INDEX']
    # exo_key = ['HSI', 'JPYKRW', 'SPX', 'USDKRW', 'EM_INDEX']
    exo_key = ['HSI', 'JPYKRW', 'SPX', 'DOLLAR_INDEX', 'EM_INDEX', 'GOLD_PRICE', 'RTY', 'OIL', 'USDKRW']
    attr_key = ['PX_HIGH', 'PX_LOW', 'PX_OPEN', 'PX_LAST']
    tot_key = []

    for a in (exo_key + endo_key):
        for b in attr_key:
            tot_key.append(a + '_' + b)

    df = data.loc[:, tot_key]
    df = df.dropna()
    for a in (endo_key + exo_key):
        vol_key = a + '_VOL'
        hi_key = a + '_PX_HIGH'
        lo_key = a + '_PX_LOW'
        df[vol_key] = df[hi_key] - df[lo_key]

    exo_list = [df[key] for key in [x + '_VOL' for x in exo_key + endo_key]]

    kospi_last = df['KOSPI_PX_LAST']
    kospi_open = df['KOSPI_PX_OPEN']
    kospi_low = df['KOSPI_PX_LOW']
    kospi_high = df['KOSPI_PX_HIGH']
    df['KOSPI_TREND'] = (kospi_last - kospi_open)/(kospi_high - kospi_low)

    # df_np = np.array([df_v, df_t] + exo_list)
    # df_np = df_np.transpose()

    df_y = np.array([df['KOSPI_VOL'], df['KOSPI_TREND']]).transpose()
    df_x = np.array(exo_list).transpose()

    # df_y = df_y[1:, :]
    # df_x = df_x[:-1, :]

    # loof_back = df_x.shape[1]
    loof_back = 5
    iteration = 50

    # Data preprocessing
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.values
    df = df.reshape(-1, 1)
    df = scaler.fit_transform(df)
    '''

    data_x, data_y = [], []
    for i in range(len(df_x) - loof_back - 1):
        data_x.append(df_x[i:(i+loof_back)])
        data_y.append(df_y[i+loof_back+1])

    df_x = data_x
    print(df_x)
    df_y = data_y

    df_y = (df_y - np.mean(df_y, axis=0)) / np.std(df_y, axis=0)
    df_x = (df_x - np.mean(df_x, axis=0)) / np.std(df_x, axis=0)

    ratio_8 = int(len(df_y)*0.8)
    ratio_9 = int(len(df_y)*0.9)

    x_train = df_x[:ratio_8]
    x_val = df_x[ratio_8:ratio_9]
    x_test = df_x[ratio_9:]

    y_train = df_y[:ratio_8]
    y_val = df_y[ratio_8:ratio_9]
    y_test = df_y[ratio_9:]

    # Feature data preprocessing
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Model designing
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(1, loof_back, len(exo_key) + len(endo_key))))
    model.add(Dropout(0.3))
    model.add(Dense(2))

    # Set Learning config
    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    # Create TensorBoard callback
    tb_hist = TensorBoard(log_dir='./graph/simple_lstm_normalscaling_attr2_mv_lag{}_iter{}'.format(loof_back, iteration), histogram_freq=0, write_grads=True, write_images=True)

    hist = model.fit(x_train, y_train, epochs=iteration,  batch_size=1, validation_data=(x_val, y_val), callbacks=[tb_hist])

    plt.plot(hist.history['loss'], 'y', label='train loss')
    plt.plot(hist.history['val_loss'], 'r', label='val loss')

    model.save('./model/simple_lstm_normalscaling_attr2_mv_lag{}_iter{}.h5'.format(loof_back, iteration))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # Model Evaluation
    trainScore = model.evaluate(x_train, y_train, verbose=0)
    model.reset_states()
    print('Train Score: ', trainScore)
    valScore = model.evaluate(x_val, y_val, verbose=0)
    model.reset_states()
    print('Validation Score: ', valScore)
    testScore = model.evaluate(x_test, y_test, verbose=0)
    model.reset_states()
    print('Test Score: ', testScore)

    plt.savefig('./graph/simple_normsc_mv_lag{}_iter{}'.format(loof_back, iteration))


def model_use():
    model_path = r'./model/simple_lstm_normalscaling_model_lag5_iter50.h5'

    model = load_model(model_path)
    data_path = r'C:\Users\CJ\workspace\Computational_finance'
    data_path = os.path.join(data_path, 'total_merged.xlsx')
    data = pd.read_excel(data_path)
    kospi_last = data['KOSPI_PX_LAST']
    kospi_open = data['KOSPI_PX_OPEN']
    kospi_low = data['KOSPI_PX_LOW']
    kospi_high = data['KOSPI_PX_HIGH']
    data['Volatility'] = kospi_high - kospi_low
    df_ori = data['Volatility'].dropna()

    loof_back = 5

    # Data preprocessing
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.values
    df = df.reshape(-1, 1)
    df = scaler.fit_transform(df)
    '''
    df = df_ori.values
    df = df.reshape(-1, 1)
    df = (df - np.mean(df)) / np.std(df)

    train = df[:int(len(df)*0.8)]
    val = df[int(len(df)*0.8):int(len(df)*0.9)]
    test = df[int(len(df)*0.9):]

    x_train, y_train = create_dataset_np(train, loof_back)
    x_val, y_val = create_dataset_np(val, loof_back)
    x_test, y_test = create_dataset_np(test, loof_back)

    # Feature data preprocessing
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    look_ahead = len(y_test)
    x_hat = x_test[0]
    predictions = np.zeros((look_ahead, 1))

    for i in range(look_ahead):
        predicition = model.predict(np.array([x_hat]), batch_size=1)
        predictions[i] = predicition
        x_hat = np.vstack([x_hat[1:], y_test[i]])

    plt.figure(figsize=(12, 5))
    plt.plot(df_ori.index[-look_ahead:], predictions, 'r-_', label="prediction")
    # plt.plot(np.arange(look_ahead), y_train[:look_ahead], label="test function")
    plt.plot(df_ori.index[-look_ahead:], y_test, 'b-_', label="original")

    mse_error = np.sqrt(np.average(((predicition - y_test)**2)))
    print(mse_error)

    plt.legend()
    plt.show()


def model_use_2atr():
    # Best
    model_path = r'./model/simple_lstm_normalscaling_attr2_lag5_iter50.h5'
    # model_path = r'./model/state_lstm_normsc_2atr_lag5_iter50.h5'
    # model_path = r'./model/stack_lstm_normsc_2atr_lag5_iter25.h5'

    model = load_model(model_path)
    data_path = r'C:\Users\CJ\workspace\Computational_finance'
    data_path = os.path.join(data_path, 'total_merged.xlsx')
    data = pd.read_excel(data_path)

    endo_key = ['KOSPI']
    exo_key = ['HSI', 'JPYKRW', 'SPX', 'DOLLAR_INDEX', 'EM_INDEX', 'GOLD_PRICE', 'RTY', 'OIL', 'USDKRW']
    attr_key = ['PX_HIGH', 'PX_LOW', 'PX_OPEN', 'PX_LAST']
    tot_key = []
    test_key = ['CALL', 'PUT', 'VKOSPI']
    tot_key = []
    bt_key = []

    for a in (exo_key + endo_key):
        for b in attr_key:
            tot_key.append(a + '_' + b)

    # bt_key = tot_key + test_key
    for a in (endo_key + test_key):
        for b in attr_key:
            bt_key.append(a + '_' + b)

    kospi_last = data['KOSPI_PX_LAST']
    kospi_open = data['KOSPI_PX_OPEN']
    kospi_low = data['KOSPI_PX_LOW']
    kospi_high = data['KOSPI_PX_HIGH']
    data['Volatility'] = kospi_high - kospi_low
    data['Trend'] = (kospi_last - kospi_open) / (kospi_high - kospi_low)
    df_v = data['Volatility'].dropna()
    df_t = data['Trend'].dropna()
    df = np.array([df_v, df_t])
    df = df.transpose()

    df_backtest = data.loc[:, bt_key]
    df_backtest = df_backtest.dropna()
    kospi_last = df_backtest['KOSPI_PX_LAST']
    kospi_open = df_backtest['KOSPI_PX_OPEN']
    kospi_low = df_backtest['KOSPI_PX_LOW']
    kospi_high = df_backtest['KOSPI_PX_HIGH']
    df_backtest['KOSPI_VOL'] = kospi_high - kospi_low
    df_backtest['KOSPI_TREND'] = (kospi_last - kospi_open) / (kospi_high - kospi_low)


    # test_list = [df_backtest[key] for key in [x + '_VOL' for x in endo_key]] + [df_backtest[key] for key in [x + '_TREND' for x in endo_key]]
    # df_backtest_y = np.array([df_backtest['KOSPI_VOL'], df_backtest['KOSPI_TREND']]).transpose()
    df_backtest_x = np.array([df_backtest['KOSPI_VOL'], df_backtest['KOSPI_TREND']]).transpose()
    # df_backtest_x = np.array(test_list).transpose()
    loof_back = 5

    # Data preprocessing
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.values
    df = df.reshape(-1, 1)
    df = scaler.fit_transform(df)
    '''

    # df_backtest_x, df_backtest_y = create_dataset_np(df_backtest, loof_back)

    df_mean = np.mean(df, axis=0)
    df_std = np.std(df, axis=0)

    df = (df - np.mean(df)) / np.std(df)
    df_backtest_x = (df_backtest_x - df_mean) / df_std

    train = df[:int(len(df)*0.8)]
    val = df[int(len(df)*0.8):int(len(df)*0.9)]
    test = df[int(len(df)*0.9):]
    x_backtest = df_backtest_x

    x_train, y_train = create_dataset_np(train, loof_back)
    x_val, y_val = create_dataset_np(val, loof_back)
    x_test, y_test = create_dataset_np(test, loof_back)
    x_backtest, y_backtest = create_dataset_np(df_backtest_x, loof_back)

    # Feature data preprocessing
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    x_backtest = np.reshape(x_backtest, (x_backtest.shape[0], x_backtest.shape[1], x_backtest.shape[2]))

    look_ahead = len(y_test)
    x_hat = x_test[0]
    predictions = np.zeros((look_ahead, 2))

    # MSE Error
    for i in range(look_ahead):
        predicition = model.predict(np.array([x_hat]), batch_size=1)
        predictions[i] = predicition
        x_hat = np.vstack([x_hat[1:], y_test[i]])

    plt.figure(figsize=(12, 5))
    plt.plot(df_v.index[-look_ahead:], predictions[:, 0], 'r-_', label="prediction")
    # plt.plot(np.arange(look_ahead), y_train[:look_ahead], label="test function")
    plt.plot(df_v.index[-look_ahead:], y_test[:, 0], 'b-_', label="test function")

    mse_error = np.sqrt(np.average(((predictions[:, 0] - y_test[:, 0])**2)))
    print(mse_error)

    plt.legend()
    plt.show()

    # Bias and Trend test
    trend = 1
    bias = 0
    N = len(predictions)
    mo = LinearRegression(fit_intercept=True)
    mo = mo.fit(predictions[:,], y_test[:,])
    print(mo.coef_)
    print(mo.intercept_)

    #trend_test = stats.ttest_1samp(predictions[:, 0])
    # Sign Error
    a = predictions[:, 0]*y_test[:, 0] > 0
    print(N, a.sum())

    # Back testing
    look_ahead = len(y_backtest)
    x_hat = x_backtest[0]
    predictions = np.zeros((look_ahead, 2))

    for i in range(look_ahead):
        predicition = model.predict(np.array([x_hat]), batch_size=1)
        predictions[i] = predicition
        x_hat = np.vstack([x_hat[1:], y_backtest[i]])


    for i in range(len(y_backtest)):
        predicition = model.predict(np.array([x_backtest[i]]), batch_size=1)
        predictions[i] = predicition
    pred_vol = predictions[:, 0]*df_std[0] + df_mean[0]
    i_vol = df_backtest['VKOSPI_PX_OPEN'] / np.sqrt(30)
    i_vol = i_vol[loof_back + 1:]

    put_profit = 0
    call_profit = 0
    k = 3
    for i in range(len(x_backtest)-k):
        if i_vol[i] > pred_vol[i]:
            print('signal')
            put_profit += (df_backtest['PUT_PX_OPEN'][i] - df_backtest['PUT_PX_OPEN'][i+k])
            call_profit += (df_backtest['CALL_PX_OPEN'][i] - df_backtest['CALL_PX_OPEN'][i+k])

    print(put_profit)
    print(call_profit)


def model_use_2atr_mv():
    model_path = r'./model/simple_lstm_model.h5'
    model_path = r'./model/stack_lstm_model.h5'
    model_path = r'./model/stack_lstm_model_lag30_iter30.h5'
    model_path = r'./model/simple_lstm_model_lag8_iter200.h5'
    model_path = r'./model/state_lstm_normsc_2atr_mv_lag5_iter30.h5'

    model = load_model(model_path)
    data_path = r'C:\Users\CJ\workspace\Computational_finance'
    data_path = os.path.join(data_path, 'total_merged.xlsx')
    data = pd.read_excel(data_path)

    endo_key = ['KOSPI']
    # exo_key = ['GOLD_PRICE', 'HSI', 'JPYKRW', 'NASDAQ', 'OIL', 'RTY', 'SHI', 'SOX', 'SPX', 'USDKRW', 'EM_INDEX']
    exo_key = ['HSI', 'JPYKRW', 'SPX', 'DOLLAR_INDEX', 'EM_INDEX', 'GOLD_PRICE', 'RTY', 'OIL', 'USDKRW']
    attr_key = ['PX_HIGH', 'PX_LOW', 'PX_OPEN', 'PX_LAST']
    tot_key = []
    test_key = ['CALL', 'PUT', 'VKOSPI']
    tot_key = []
    bt_key = []

    for a in (exo_key + endo_key):
        for b in attr_key:
            tot_key.append(a + '_' + b)

    # bt_key = tot_key + test_key
    for a in (exo_key + endo_key + test_key):
        for b in attr_key:
            bt_key.append(a + '_' + b)

    df = data.loc[:, tot_key]
    df = df.dropna()

    df_backtest = data.loc[:, bt_key]
    df_backtest = df_backtest.dropna()

    for a in (endo_key + exo_key):
        vol_key = a + '_VOL'
        hi_key = a + '_PX_HIGH'
        lo_key = a + '_PX_LOW'
        df[vol_key] = df[hi_key] - df[lo_key]
        df_backtest[vol_key] = df_backtest[hi_key] - df_backtest[lo_key]

    exo_list = [df[key] for key in [x + '_VOL' for x in exo_key + endo_key]]
    test_list = [df_backtest[key] for key in [x + '_VOL' for x in exo_key + endo_key]]

    kospi_last = df['KOSPI_PX_LAST']
    kospi_open = df['KOSPI_PX_OPEN']
    kospi_low = df['KOSPI_PX_LOW']
    kospi_high = df['KOSPI_PX_HIGH']
    df['KOSPI_TREND'] = (kospi_last - kospi_open)/(kospi_high - kospi_low)
    df_backtest['KOSPI_TREND'] = (kospi_last - kospi_open)/(kospi_high - kospi_low)

    # df_np = np.array([df_v, df_t] + exo_list)
    # df_np = df_np.transpose()

    df_y = np.array([df['KOSPI_VOL'], df['KOSPI_TREND']]).transpose()
    df_x = np.array(exo_list).transpose()

    df_backtest_y = np.array([df_backtest['KOSPI_VOL'], df_backtest['KOSPI_TREND']]).transpose()
    df_backtest_x = np.array(test_list).transpose()

    # df_y = df_y[1:, :]
    # df_x = df_x[:-1, :]

    # loof_back = df_x.shape[1]
    loof_back = 5
    iteration = 30

    # Data preprocessing
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.values
    df = df.reshape(-1, 1)
    df = scaler.fit_transform(df)
    '''

    data_x, data_y = [], []
    for i in range(len(df_x) - loof_back - 1):
        data_x.append(df_x[i:(i + loof_back)])
        data_y.append(df_y[i + loof_back + 1])

    df_x = data_x
    df_y = data_y

    data_x, data_y = [], []
    for i in range(len(df_backtest_x) - loof_back - 1):
        data_x.append(df_backtest_x[i:(i + loof_back)])
        data_y.append(df_backtest_y[i + loof_back + 1])

    df_backtest_x = data_x
    df_backtest_y = data_y
    df_y_mean = np.mean(df_y, axis=0)
    df_y_std = np.std(df_y, axis=0)
    df_x_mean = np.mean(df_x, axis=0)
    df_x_std = np.std(df_x, axis=0)

    df_y = (df_y - np.mean(df_y, axis=0)) / np.std(df_y, axis=0)
    df_x = (df_x - np.mean(df_x, axis=0)) / np.std(df_x, axis=0)

    df_backtest_y = (df_backtest_y - df_y_mean) / df_y_std
    df_backtest_x = (df_backtest_x - df_x_mean) / df_x_std

    ratio_8 = int(len(df_y) * 0.8)
    ratio_9 = int(len(df_y) * 0.9)

    x_train = df_x[:ratio_8]
    x_val = df_x[ratio_8:ratio_9]
    x_test = df_x[ratio_9:]
    x_backtest = df_backtest_x

    y_train = df_y[:ratio_8]
    y_val = df_y[ratio_8:ratio_9]
    y_test = df_y[ratio_9:]
    y_backtest = df_backtest_y

    for a in (exo_key + endo_key):
        for b in attr_key:
            tot_key.append(a + '_' + b)

    df = data.loc[:, tot_key]
    df = df.dropna()
    for a in (endo_key + exo_key):
        vol_key = a + '_VOL'
        hi_key = a + '_PX_HIGH'
        lo_key = a + '_PX_LOW'
        df[vol_key] = df[hi_key] - df[lo_key]

    exo_list = [df[key] for key in [x + '_VOL' for x in exo_key + endo_key]]

    kospi_last = df['KOSPI_PX_LAST']
    kospi_open = df['KOSPI_PX_OPEN']
    kospi_low = df['KOSPI_PX_LOW']
    kospi_high = df['KOSPI_PX_HIGH']
    df['KOSPI_TREND'] = (kospi_last - kospi_open)/(kospi_high - kospi_low)

    # df_np = np.array([df_v, df_t] + exo_list)
    # df_np = df_np.transpose()

    df_y = np.array([df['KOSPI_VOL'], df['KOSPI_TREND']]).transpose()
    df_x = np.array(exo_list).transpose()

    # loof_back = df_x.shape[1]
    loof_back = 5
    iteration = 50

    # Data preprocessing
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.values
    df = df.reshape(-1, 1)
    df = scaler.fit_transform(df)
    '''

    data_x, data_y = [], []
    for i in range(len(df_x) - loof_back - 1):
        data_x.append(df_x[i:(i + loof_back)])
        data_y.append(df_y[i + loof_back + 1])

    df_x = data_x
    df_y = data_y

    df_y = (df_y - np.mean(df_y, axis=0)) / np.std(df_y, axis=0)
    df_x = (df_x - np.mean(df_x, axis=0)) / np.std(df_x, axis=0)

    ratio_8 = int(len(df_y)*0.8)
    ratio_9 = int(len(df_y)*0.9)

    x_train = df_x[:ratio_8]
    x_val = df_x[ratio_8:ratio_9]
    x_test = df_x[ratio_9:]

    y_train = df_y[:ratio_8]
    y_val = df_y[ratio_8:ratio_9]
    y_test = df_y[ratio_9:]

    # Feature data preprocessing
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    look_ahead = len(y_test)
    # x_hat = x_test[0]
    predictions = np.zeros((look_ahead, 2))

    for i in range(look_ahead):
        predicition = model.predict(np.array([x_test[i]]), batch_size=1)
        predictions[i] = predicition

    plt.figure(figsize=(12, 5))
    plt.plot(df.index[-look_ahead:], predictions[:, 0], 'r-_', label="prediction")
    # plt.plot(np.arange(look_ahead), y_train[:look_ahead], label="test function")
    plt.plot(df.index[-look_ahead:], y_test[:, 0], 'b-_', label="test function")
    # plt.plot(df.index[-look_ahead:], predictions[:, 0] - y_test[:, 0], 'g-_', label="difference")

    mse_error = np.sqrt(np.average(((predictions[:, 0] - y_test[:, 0])**2)))
    print(mse_error)

    plt.legend()
    plt.show()


def model_use_2atr_mv_lag():
    model_path = r'./model/simple_lstm_normalscaling_attr2_mv_lag5_iter50.h5'
    model = load_model(model_path)

    data_path = r'C:\Users\CJ\workspace\Computational_finance'
    # data_path = os.path.join(data_path, 'Data_set.csv')
    data_path = os.path.join(data_path, 'total_merged.xlsx')
    data = pd.read_excel(data_path)

    endo_key = ['KOSPI']
    # exo_key = ['GOLD_PRICE', 'HSI', 'JPYKRW', 'NASDAQ', 'OIL', 'RTY', 'SHI', 'SOX', 'SPX', 'USDKRW', 'EM_INDEX']
    # exo_key = ['HSI', 'JPYKRW', 'SPX', 'USDKRW', 'EM_INDEX']
    exo_key = ['HSI', 'JPYKRW', 'SPX', 'DOLLAR_INDEX', 'EM_INDEX', 'GOLD_PRICE', 'RTY', 'OIL', 'USDKRW']
    attr_key = ['PX_HIGH', 'PX_LOW', 'PX_OPEN', 'PX_LAST']
    # test_key = ['CALL_PX_OPEN', 'CALL_PX_LAST', 'PUT_PX_OPEN', 'PUT_PX_LAST', 'VKOSPI_OPEN']
    test_key = ['CALL', 'PUT', 'VKOSPI']
    tot_key = []
    bt_key = []

    for a in (exo_key + endo_key):
        for b in attr_key:
            tot_key.append(a + '_' + b)

    # bt_key = tot_key + test_key
    for a in (exo_key + endo_key + test_key):
        for b in attr_key:
            bt_key.append(a + '_' + b)

    df = data.loc[:, tot_key]
    df = df.dropna()

    df_backtest = data.loc[:, bt_key]
    df_backtest = df_backtest.dropna()

    for a in (endo_key + exo_key):
        vol_key = a + '_VOL'
        hi_key = a + '_PX_HIGH'
        lo_key = a + '_PX_LOW'
        df[vol_key] = df[hi_key] - df[lo_key]
        df_backtest[vol_key] = df_backtest[hi_key] - df_backtest[lo_key]

    exo_list = [df[key] for key in [x + '_VOL' for x in exo_key + endo_key]]
    test_list = [df_backtest[key] for key in [x + '_VOL' for x in exo_key + endo_key]]

    kospi_last = df['KOSPI_PX_LAST']
    kospi_open = df['KOSPI_PX_OPEN']
    kospi_low = df['KOSPI_PX_LOW']
    kospi_high = df['KOSPI_PX_HIGH']
    df['KOSPI_TREND'] = (kospi_last - kospi_open)/(kospi_high - kospi_low)
    df_backtest['KOSPI_TREND'] = (kospi_last - kospi_open)/(kospi_high - kospi_low)

    # df_np = np.array([df_v, df_t] + exo_list)
    # df_np = df_np.transpose()

    df_y = np.array([df['KOSPI_VOL'], df['KOSPI_TREND']]).transpose()
    df_x = np.array(exo_list).transpose()

    df_backtest_y = np.array([df_backtest['KOSPI_VOL'], df_backtest['KOSPI_TREND']]).transpose()
    df_backtest_x = np.array(test_list).transpose()

    # df_y = df_y[1:, :]
    # df_x = df_x[:-1, :]

    # loof_back = df_x.shape[1]
    loof_back = 5
    iteration = 30

    # Data preprocessing
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.values
    df = df.reshape(-1, 1)
    df = scaler.fit_transform(df)
    '''

    data_x, data_y = [], []
    for i in range(len(df_x) - loof_back - 1):
        data_x.append(df_x[i:(i + loof_back)])
        data_y.append(df_y[i + loof_back + 1])

    df_x = data_x
    df_y = data_y

    data_x, data_y = [], []
    for i in range(len(df_backtest_x) - loof_back - 1):
        data_x.append(df_backtest_x[i:(i + loof_back)])
        data_y.append(df_backtest_y[i + loof_back + 1])

    df_backtest_x = data_x
    df_backtest_y = data_y
    df_y_mean = np.mean(df_y, axis=0)
    df_y_std = np.std(df_y, axis=0)
    df_x_mean = np.mean(df_x, axis=0)
    df_x_std = np.std(df_x, axis=0)

    df_y = (df_y - np.mean(df_y, axis=0)) / np.std(df_y, axis=0)
    df_x = (df_x - np.mean(df_x, axis=0)) / np.std(df_x, axis=0)

    df_backtest_y = (df_backtest_y - df_y_mean) / df_y_std
    df_backtest_x = (df_backtest_x - df_x_mean) / df_x_std

    ratio_8 = int(len(df_y) * 0.8)
    ratio_9 = int(len(df_y) * 0.9)

    x_train = df_x[:ratio_8]
    x_val = df_x[ratio_8:ratio_9]
    x_test = df_x[ratio_9:]
    x_backtest = df_backtest_x

    y_train = df_y[:ratio_8]
    y_val = df_y[ratio_8:ratio_9]
    y_test = df_y[ratio_9:]
    y_backtest = df_backtest_y

    # Feature data preprocessing
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    x_backtest = np.reshape(x_backtest, (x_backtest.shape[0], x_backtest.shape[1], x_backtest.shape[2]))

    look_ahead = len(y_test)
    # x_hat = x_test[0]
    predictions = np.zeros((look_ahead, 2))

    for i in range(look_ahead):
        predicition = model.predict(np.array([x_test[i]]), batch_size=1)
        predictions[i] = predicition

    plt.figure(figsize=(12, 5))
    plt.plot(df.index[-look_ahead:], predictions[:, 0], 'r-_', label="prediction")
    # plt.plot(np.arange(look_ahead), y_train[:look_ahead], label="test function")
    plt.plot(df.index[-look_ahead:], y_test[:, 0], 'b-_', label="test function")
    # plt.plot(df.index[-look_ahead:], predictions[:, 0] - y_test[:, 0], 'g-_', label="difference")

    mse_error = np.sqrt(np.average(((predictions[:, 0] - y_test[:, 0])**2)))
    print(mse_error)

    plt.legend()
    plt.show()

    # Bias and Trend test
    trend = 1
    bias = 0
    N = len(predictions)
    mo = LinearRegression(fit_intercept=True)
    mo = mo.fit(predictions[:, ], y_test[:, ])
    print(mo.coef_)
    print(mo.intercept_)

    # trend_test = stats.ttest_1samp(predictions[:, 0])
    # Sign Error
    a = predictions[:, 0] * y_test[:, 0] > 0
    print(N, a.sum())


    for i in range(len(y_backtest)):
        predicition = model.predict(np.array([x_backtest[i]]), batch_size=1)
        predictions[i] = predicition
    pred_vol = predictions[:, 0]*df_y_std[0] + df_y_mean[0]
    i_vol = df_backtest['VKOSPI_PX_OPEN'] / np.sqrt(30)
    i_vol = i_vol[loof_back + 1:]

    put_profit = 0
    call_profit = 0
    for i in range(len(x_backtest)-1):
        if i_vol[i] > pred_vol[i]:
            print('signal')
            put_profit += (df_backtest['PUT_PX_OPEN'][i] - df_backtest['PUT_PX_OPEN'][i+1])
            call_profit += (df_backtest['CALL_PX_OPEN'][i] - df_backtest['CALL_PX_OPEN'][i+1])

    print(put_profit)
    print(call_profit)


if __name__ == '__main__':
    # simple_lstm()
    # simple_lstm_2atr()
    # simple_lstm_2atr_bn()
    # simple_lstm_2atr_mv()
    # state_lstm()
    # state_lstm_2atr()
    # state_lstm_2atr_mv()
    # stack_lstm()
    # stack_lstm_2atr()
    # stack_lstm_2atr_mv()
    # model_use()
    model_use_2atr()
    # model_use_2atr_mv()
    # model_use_2atr_mv_lag()