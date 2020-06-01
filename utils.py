import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

# 로그수익률 계산
def logret(X):
    log_ret = np.zeros_like(X)
    log_ret[0] = 0
    for i in range(1, X.shape[0]):
        log_ret[i] = math.log(X[i] / X[i-1])
    return log_ret

# Data Processing (그룹화)
def data_process(X, train_size, num_steps):
    X_result = X[:num_steps, :, :]  # [30, 1, 3]
    Y_result = X[num_steps, :, :]   #     [1, 3]
    X_last = X[num_steps-1, :, :]
    for s in range(1, train_size - num_steps):
        X_result = torch.cat((X_result, X[s : s + num_steps, :, :]), dim = 1)
        X_last = torch.cat((X_last, X[s + num_steps - 1, :, :]), dim = 0)
        Y_result = torch.cat((Y_result, X[s + num_steps, :, :]), dim = 0)
    return X_result, Y_result


def binary_result(t):
    val = t.numpy()[0][0]
    if val >= 0:
        return 1
    else:
        return 0


def data_process_bin(X, train_size, num_steps):
    y_target=[]                                # kfiri 수정 // 이전 diff_result=[]  
    X_result = X[:num_steps, :, :]  # [30, 1, 1]
    X_last = X[num_steps-1, :, :]
    Y_result = X[num_steps, :, :]   #     [1, 1]
    y_target.append(binary_result(Y_result))    # kfiri 수정 // 이전 diff_result.append(binary_result(X_last-Y_result))
    for s in range(1, train_size - num_steps):
        X_result = torch.cat((X_result, X[s : s + num_steps, :, :]), dim = 1)
        #x_last = X[s + num_steps - 1, :, :]       # kfiri 수정
        y_result = X[s + num_steps, :, :]
        X_last = torch.cat((X_last, X[s + num_steps - 1, :, :]), dim = 0)   # kfiri 수정 // X_last = torch.cat((X_last, x_last), dim = 0)
        #Y_result = torch.cat((Y_result, X[s + num_steps, :, :]), dim = 0)
        y_target.append(binary_result(y_result))                         # kfiri 수정 // diff_result.append(binary_result(y_result-x_last))
    Y_target = torch.FloatTensor(y_target)                               # kfiri 수정 // Diff_result = torch.FloatTensor(diff_result)
    return X_result, Y_target                               # kfiri 수정 // return X_result, Y_result, Diff_result  


def plot(axislengths, prices, colors, xLabels, yLabels, Title, Legends):
    plt.figure()
    for i in range(0, len(axislengths)):
        length = axislengths[i]
        plt.plot(range(0, length), prices[i][:length], color = colors[i], label = Legends[i])
    legend = plt.legend(loc='upper left')
    legend.get_frame()

    plt.xlabel(xLabels)
    plt.ylabel(yLabels)
    plt.title(Title)
    plt.show()


def visdom_graph(vis, axislengths, prices, colors, xLabels, yLabels, Title, Legends):
    # 0: prediction , 1: groundTruth 
    
    list_total = []
    list_pp = []
    accum_pp = 0
    for pp in prices[0]:
        accum_pp += pp
        list_pp.append(accum_pp*250)

    list_pg = []
    accum_pg = 0
    for pg in prices[1]:
        accum_pg += pg
        list_pg.append(accum_pg*250)

    tensor_final = torch.cat((torch.unsqueeze(torch.FloatTensor(list_pp), 0), 
                            torch.unsqueeze(torch.FloatTensor(list_pg), 0)), 0).transpose(0,1)

    price_t = torch.FloatTensor(prices).transpose(0,1)
    vis.line(X=torch.Tensor(list(range(len(prices[0])))),
            Y=tensor_final,
            opts=dict(title=Title,
                    xlabel=xLabels,
                    ylabel=yLabels,
                    legend=Legends,
                    showlegend=True)
                    )
