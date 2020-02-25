import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--dataset', type=str, default='AAPL')
parser.add_argument('--model', type=str, default='LSTM')
opt = parser.parse_args()

print(opt)