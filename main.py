import sys
from train_test import TrainSet
from evaluate import EvalSet
from option import opt

if not opt.test:
    print('Train Mode')
    Trainer = TrainSet(dataset = opt.dataset, LogReturn = opt.type)
    Trainer(opt.model)
else:
    print('Test Mode')
    Evaluator = EvalSet(dataset = opt.dataset, LogReturn = opt.type)
    Evaluator(opt.model)

