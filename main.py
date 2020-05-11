import sys
from train import TrainSet
from evaluate import EvalSet
from train_bin import TrainSetBinary
from evaluate_bin import EvalSetBinary
from option import opt

if opt.binary:
    if not opt.test:
        print('Train Mode : Binary Classificaion')
        TrainerBin = TrainSetBinary(dataset = opt.dataset, window_size= 1, LogReturn = opt.type)
        TrainerBin(opt.model)
    else:
        print('Test Mode : Binary Classificaion')
        EvaluatorBin = EvalSetBinary(dataset = opt.dataset, window_size= 1, LogReturn = opt.type)
        EvaluatorBin(opt.model)
else:
    if not opt.test:
        print('Train Mode : Regression')
        TrainerReg = TrainSet(dataset = opt.dataset, window_size= 3, LogReturn = opt.type)
        TrainerReg(opt.model)
    else:
        print('Test Mode : Regression')
        EvaluatorReg = EvalSet(dataset = opt.dataset, window_size= 3, LogReturn = opt.type)
        EvaluatorReg(opt.model)

