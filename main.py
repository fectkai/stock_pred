import sys
from train import TrainSet
from evaluate import EvalSet
from option import opt

filename = opt.dataset + '.csv'

if opt.mode == 'train':
    Trainer = TrainSet(filename, LogReturn = True)
    Trainer(opt.model)
else:
    Evaluator = EvalSet(filename, LogReturn = True)
    Evaluator(opt.model)
