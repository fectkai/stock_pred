import sys
from train import TrainSet
from evaluate import EvalSet
from train_bin import TrainSetBinary
from evaluate_bin import EvalSetBinary
from option import opt

if opt.binary:
    if not opt.test:
        print('Train Mode : Binary Classificaion')
        Trainer = TrainSet(dataset = opt.dataset, window_size= 1, LogReturn = opt.type)
        Trainer(opt.model)
    else:
        print('Test Mode : Binary Classificaion')
        Evaluator = EvalSet(dataset = opt.dataset, window_size= 1, LogReturn = opt.type)
        Evaluator(opt.model)
else:
    if not opt.test:
        print('Train Mode : Regression')
        Trainer = TrainSet(dataset = opt.dataset, window_size= 3, LogReturn = opt.type)
        Trainer(opt.model)
    else:
        print('Test Mode : Regression')
        Evaluator = EvalSet(dataset = opt.dataset, window_size= 3, LogReturn = opt.type)
        Evaluator(opt.model)

