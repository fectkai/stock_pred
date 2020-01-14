import sys
from train import TrainSet
from evaluate import EvalSet

ticker = sys.argv[2] + '.csv'

# python3 data_cleaning.py AAPL
# python3 main.py train AAPL LSTM
# python3 main.py test AAPL LSTM

if sys.argv[1] == 'train':
    Trainer = TrainSet(ticker, LogReturn = True)
    Trainer(sys.argv[3])
else:
    Evaluator = EvalSet(ticker, LogReturn = True)
    Evaluator(sys.argv[3])
