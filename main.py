import sys
from train import TrainSet
from evaluate import EvalSet
from option import opt

if not opt.test:
    Trainer = TrainSet(dataset = opt.dataset, LogReturn = opt.type)
    Trainer(opt.model)
else:
    Evaluator = EvalSet(dataset = opt.dataset, LogReturn = opt.type)
    Evaluator(opt.model)

