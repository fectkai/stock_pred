import numpy as np
import torch
from torch import nn

from data_loading import Loader
import utils
from option import opt
import visdom

class EvalSet:
    def __init__(self, dataset, window_size = 3, LogReturn = 'log'):
        self.dataset = dataset
        self.filename = dataset + '.csv'
        self.prices = Loader(self.filename, window_size, LogReturn = LogReturn)

    def __call__(self, modelname, split_rate = .9, seq_length = 30, 
                                                batch_size = 8, num_layers = 2):
        vis = visdom.Visdom()
        train_size = int(self.prices.train_size * split_rate)
        X = self.prices.X[ train_size : train_size + 300, :]
        X = torch.unsqueeze(torch.from_numpy(X).float(), 1)
        X_test, Y_test = utils.data_process(X, X.shape[0], seq_length)
        X_test = X_test.to(opt.device)
        Y_test = Y_test.to(opt.device)
        model = torch.load('trained_model/'+modelname + '_' + self.dataset + '.model')
        model.eval()
        model = model.to(opt.device)

        loss_fn = nn.MSELoss().to(opt.device)
        with torch.no_grad():
            loss_sum = 0
            Y_pred = model(X_test[:, :batch_size, :])       # [2, b, 3]
            if batch_size==1:   
                Y_pred = torch.unsqueeze(Y_pred, 1)
            Y_pred = torch.squeeze(Y_pred[num_layers - 1, :, :])    # [b, 3]
            for i in range(batch_size, X_test.shape[1], batch_size):
                y = model(X_test[:, i : i + batch_size, :])
                if batch_size==1:   
                    y = torch.unsqueeze(y, 1)
                y = torch.squeeze(y[num_layers - 1, :, :])
                Y_pred = torch.cat((Y_pred, y))

                loss = loss_fn(Y_test[i : i + batch_size, :], y)
                loss_sum += loss.item()

        print(loss_sum)
        Y_pred.resize_(Y_pred.shape[0] * Y_pred.shape[1])
        Y_test.resize_(Y_test.shape[0] * Y_test.shape[1])
        Y_final = torch.cat([torch.unsqueeze(Y_pred,1), torch.unsqueeze(Y_test,1)], dim=1)
        # axislengths, prices, colors, xLabels, yLabels, Title, Legends
        if opt.debug_mode:
            utils.plot(
                [Y_pred.shape[0], Y_test.shape[0]], 
                [Y_pred.cpu().numpy(), Y_test.cpu().numpy()], 
                ['blue', 'red'],
                'Time (Days)',
                'Price',
                'Sample ' + modelname + ' Price Result',
                ['Prediction', 'Ground Truth'])
        else:
            
            vis.line(X= torch.Tensor(list(range(len(Y_pred)))),
                    Y=Y_final,
                    opts=dict(title=opt.dataset + ' dataset ' + opt.model + ' ' + opt.type + ' Result',
                            xlabel='Time (Days)',
                            ylabel=opt.type,
                            legend=['Prediction', 'Ground Truth'],
                            showlegend=True)
                    )
            '''
            utils.visdom_graph(vis,
                [Y_pred.shape[0], Y_test.shape[0]], 
                [Y_pred.cpu().numpy(), Y_test.cpu().numpy()], 
                ['blue', 'red'],
                'Time (Days)',
                'Price',
                opt.dataset + ' dataset ' + opt.model + ' Price Result',
                ['Prediction', 'Ground Truth'])
            '''
            


if __name__=="__main__":
    Evaluator = EvalSet(opt.dataset, LogReturn = opt.type)
    Evaluator(opt.model)