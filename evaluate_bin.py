import numpy as np
import torch
from torch import nn

from data_loading import Loader
import utils
from option import opt
import visdom

class EvalSetBinary:
    def __init__(self, dataset, window_size = 1, LogReturn = 'log'):
        self.dataset = dataset
        self.filename = dataset + '.csv'
        self.prices = Loader(self.filename, window_size, LogReturn = LogReturn)

    def __call__(self, modelname, split_rate = .9, seq_length = 30, 
                                                batch_size = 8, num_layers = 2):
        vis = visdom.Visdom()
        train_size = int(self.prices.train_size * split_rate)
        X = self.prices.X[ train_size : train_size + 300, :]
        X = torch.unsqueeze(torch.from_numpy(X).float(), 1)
        X_test, Y_test, diff_test = utils.data_process_bin(X, X.shape[0], seq_length)
        X_test = X_test.to(opt.device)
        Y_test = Y_test.to(opt.device)
        diff_test = diff_test.to(opt.device)

        model = torch.load('trained_model/'+modelname + '_' + self.dataset + '_bin.model')
        model.eval()
        model = model.to(opt.device)

        loss_fn = nn.BCELoss().to(opt.device)

        with torch.no_grad():
            loss_sum = 0
            Y_pred = model(X_test[:, :batch_size, :])       # [2, b, 3]
            Y_pred = torch.unsqueeze(Y_pred, 2)
            Y_pred = torch.squeeze(Y_pred[num_layers - 1, :, :])    # [b, 3]
            for i in range(batch_size, X_test.shape[1], batch_size):
                y = model(X_test[:, i : i + batch_size, :])
                y = torch.unsqueeze(y, 2)
                y = torch.squeeze(y[num_layers - 1, :, :])
                Y_pred = torch.cat((Y_pred, y))

                loss = loss_fn(y, diff_test[i : i + batch_size])
                loss_sum += loss.item()

        print(loss_sum)
        ### ====================================================================
        Y_pred.resize_(Y_pred.shape[0] * Y_pred.shape[1])
        diff_test.resize_(diff_test.shape[0] * diff_test.shape[1])
        Y_final = torch.cat([torch.unsqueeze(Y_pred,1), torch.unsqueeze(diff_test,1)], dim=1)
            
        vis.line(X= torch.Tensor(list(range(len(Y_pred)))),
                Y=Y_final,
                opts=dict(title=opt.dataset + ' dataset ' + opt.model + ' ' + opt.type + ' Result',
                        xlabel='Time (Days)',
                        ylabel=opt.type,
                        legend=['Prediction', 'Ground Truth'],
                        showlegend=True)
                )
            


if __name__=="__main__":
    Evaluator = EvalSetBinary(opt.dataset, LogReturn = opt.type)
    Evaluator(opt.model)