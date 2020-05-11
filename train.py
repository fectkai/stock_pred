import numpy as np
import torch
from torch import nn
from torch import optim
from torch.backends import cudnn

from nets import SimpleLSTM, SimpleGRU
from data_loading import Loader
import utils
import visdom
from option import opt
import time

stock_name_ = {
    'AAPL':'Apple',
    'DIS':'Walt Disney',
    'LPL':'LG Display',
    'NSRGY':'Nestle S.A.',
    'KOSPI':'KOSPI200'
}


class TrainSet:
    def __init__(self, dataset, window_size = 3, LogReturn = 'price'):
        self.dataset = dataset
        self.filename = dataset + '.csv'
        self.prices = Loader(self.filename, window_size, LogReturn = LogReturn)
        self.window_size = window_size

    def __call__(self, model_name, hidden_size = 128, seq_length = 30, 
            split_rate = .9, num_layers = 2):
        
        vis = visdom.Visdom()
        batch_size = opt.batch_size

        train_size = int(self.prices.train_size * split_rate)
        X = torch.unsqueeze(torch.from_numpy(self.prices.X[:train_size, :]).float(), 1)
        X_train, Y_train = utils.data_process(X, train_size, seq_length)
        X_train = X_train.to(opt.device)
        Y_train = Y_train.to(opt.device)

        if model_name == 'LSTM':
            model = SimpleLSTM(self.window_size, hidden_size, num_layers = num_layers)
        else:
            model = SimpleGRU(self.window_size, hidden_size, num_layers = num_layers)

        model = model.to(opt.device)

        if opt.device=='cuda':
            model=torch.nn.DataParallel(model)
            cudnn.benchmark=True
        
        loss_fn = nn.MSELoss().to(opt.device)
        optimizer = optim.Adam(model.parameters())
        loss_plt = []

        timeStart = time.time()

        for epoch in range(opt.num_epochs):
            loss_sum = 0
            Y_pred = model(X_train[:, :batch_size, :])
            if batch_size==1:
                Y_pred = torch.unsqueeze(Y_pred, 1)
            Y_pred = torch.squeeze(Y_pred[num_layers-1, :, :])
            for i in range(batch_size, X_train.shape[1], batch_size):
                y = model(X_train[:, i : i + batch_size, :])
                if batch_size==1:
                    y = torch.squeeze(y, 1)
                y = torch.squeeze(y[num_layers - 1, :, :])
                Y_pred = torch.cat((Y_pred, y))
                loss = loss_fn(Y_train[i : i + batch_size, :], y)
                loss_sum += loss.item()

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
            
            # Visdom
            vis.line(X=torch.ones((1, 1)).cpu() * i + epoch * train_size,
                    Y=torch.Tensor([loss_sum]).unsqueeze(0).cpu(),
                    win='loss',
                    update='append',
                    opts=dict(xlabel='step',
                           ylabel='Loss',
                           title='Training Loss {} (bs={})'.format(stock_name_[self.dataset], batch_size),
                           legend=['Loss'])
                 )
            
            print('epoch [%d] finished, Loss Sum: %f' % (epoch, loss_sum))
            loss_plt.append(loss_sum)
            if epoch % 30 == 0:
                print('testing')
                with torch.no_grad():
                    a = Y_pred
                    b = Y_train
                    a = a.contiguous().view(2314*3)
                    b = b.contiguous().view(2314*3)
                    Y_final = torch.cat([torch.unsqueeze(a,1), torch.unsqueeze(b,1)], dim=1)
                    vis.line(X= torch.Tensor(list(range(len(a)))),
                            Y=Y_final,
                            win='testing',
                            opts=dict(title=opt.dataset + ' dataset ' + opt.model + ' ' + opt.type + 'Result',
                                    xlabel='Time (Days)',
                                    ylabel=opt.type,
                                    legend=['Prediction', 'Ground Truth'],
                                    showlegend=True)
                            )

        timeSpent = time.time() - timeStart
        print('Time Spend : {}'.format(timeSpent))
        torch.save(model, 'trained_model/'+model_name + '_'+ self.dataset + '.model')
        # utils.plot([len(loss_plt)], [np.array(loss_plt)], 'black', 'Epoch', 'Loss Sum', 'MSE Loss Function')
        # utils.visdom_graph(vis, [len(loss_plt)], [np.array(loss_plt)], 'black', 'Epoch', 'Loss Sum', 'MSE Loss Function')



if __name__ == "__main__":
    Trainer = TrainSet(opt.dataset, LogReturn = opt.type)
    Trainer(opt.model)