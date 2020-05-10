import numpy as np
import utils

class Loader:

    def __init__(self, filename, window_size, LogReturn):
        if filename.split('.')[0]=='KOSPI':
            print('Using KOSPI200 Data')
            adjusted_close = np.genfromtxt('data/'+filename, delimiter = ',', skip_header = 1, usecols = (4))
        else:
            adjusted_close = np.genfromtxt('data/'+filename, delimiter = ',', skip_header = 1, usecols = (5))

        if LogReturn == 'log':
            log_return = utils.logret(adjusted_close) 
        elif LogReturn == 'price':
            log_return = adjusted_close
        elif LogReturn == 'volat':
            print('volatality is not ready...')
            exit()

        self.train_size = log_return.shape[0] // window_size

        self.log_return = log_return[:self.train_size * window_size]
        self.X = self.log_return.reshape(self.train_size, window_size)


if __name__ == "__main__":
    Loader(filename='KOSPI.csv', window_size=3, LogReturn='price')