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
        period_end_size = self.prices.train_size                 #kfiri 추가
        X = self.prices.X[ train_size : period_end_size, :]      #kfiri 수정  
        X = torch.unsqueeze(torch.from_numpy(X).float(), 1)
        print(X.shape[0])
        X_test, Y_target = utils.data_process_bin(X, X.shape[0], seq_length)         #kfiri 수정 //  X_test, Y_test, diff_test = utils.data_process_bin(X, X.shape[0], seq_length)
        X_test = X_test.to(opt.device)
        # Y_test = Y_test.to(opt.device)
        Y_target = Y_target.to(opt.device)

        model = torch.load('trained_model/'+modelname + '_' + self.dataset + '_bin_' + opt.type + '.model')
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

                loss = loss_fn(y, Y_target[i : i + batch_size])                   #kfiri 수정 // loss = loss_fn(y, diff_test[i : i + batch_size])
                loss_sum += loss.item()

        Y_final = torch.cat([torch.unsqueeze(Y_pred[:30],1), torch.unsqueeze(Y_target[:30],1)], dim=1)
        # axislengths, prices, colors, xLabels, yLabels, Title, Legends
        vis.line(X= torch.Tensor(list(range(len(Y_pred[:30])))),
                Y=Y_final,
                opts=dict(title=opt.dataset + ' dataset ' + opt.model + ' ' + opt.type + ' Result (Classification)',
                        xlabel='Time (Days)',
                        ylabel=opt.type,
                        win='test_reg',
                        legend=['Prediction', 'Ground Truth'],
                        showlegend=True)
                )
        # print(loss_sum)    kfiri 전체 PP
        count_0 = 0
        count_1 = 0
        Y_target_sum_0 = 0
        Y_target_sum_1 = 0
        for i in range(Y_target.shape[0]):    #kfiri 수정 // for i in range(diff_test.shape[0]):
            if Y_target.data[i] == 0:
                Y_target_sum_0 = Y_target_sum_0 +1
                if Y_pred.data[i] < 0.5:
                    count_0 = count_0 +1
            else:
                Y_target_sum_1 = Y_target_sum_1 +1
                if Y_pred.data[i] >= 0.5:
                    count_1 = count_1 +1
        print('Total_hit :','{}%'.format(round((count_0 + count_1) / Y_target.shape[0]*100, 2)))   #kfiri 수정 // print('{}%'.format((count / diff_test.shape[0])*100))
        print('Negat_hit :','{}%'.format(round((count_0 / Y_target_sum_0)*100, 2)))   #kfiri 수정 // print('{}%'.format((count / diff_test.shape[0])*100))        
        print('Posit_hit :','{}%'.format(round((count_1 / Y_target_sum_1)*100, 2)))   #kfiri 수정 // print('{}%'.format((count / diff_test.shape[0])*100))
        print('Pred_N_True_N :',count_0)        
        print('Pred_P_True_P :',count_1)        
        print('True_N :',Y_target_sum_0)
        print('True_P :',Y_target_sum_1)
        print(len(Y_pred))             # kfiri 추가


if __name__=="__main__":
    Evaluator = EvalSetBinary(opt.dataset, LogReturn = opt.type)
    Evaluator(opt.model)