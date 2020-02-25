import sys
from train import TrainSet
from evaluate import EvalSet
from option import opt

ticker = opt.dataset + '.csv'

# python3 data_cleaning.py AAPL
# python3 main.py train AAPL LSTM
# python3 main.py test AAPL LSTM

if opt.mode == 'train':
    Trainer = TrainSet(ticker, LogReturn = True)
    Trainer(opt.model)
else:
    Evaluator = EvalSet(ticker, LogReturn = True)
    Evaluator(opt.model)
