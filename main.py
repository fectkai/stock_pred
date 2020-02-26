import sys
from train import TrainSet
from evaluate import EvalSet
from option import opt


if opt.mode == 'train':
    Trainer = TrainSet(opt.dataset, LogReturn = True)
    Trainer(opt.model)
elif opt.mode == 'test':
    Evaluator = EvalSet(opt.dataset, LogReturn = True)
    Evaluator(opt.model)
else:
    print('train or test')
