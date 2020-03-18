import argparse
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--dataset', type=str, default='AAPL')
parser.add_argument('--model', type=str, default='LSTM')
parser.add_argument('--bs', dest='batch_size', help='batch size', default=8, type=int)
parser.add_argument('--epoch', dest='num_epochs', default=500, type=int)
parser.add_argument('--dm', dest='debug_mode', action='store_true')
parser.add_argument('--tm', dest='debug_mode', action='store_false')
parser.set_defaults(debug_mode=True)
opt = parser.parse_args()

opt.device='cuda' if torch.cuda.is_available() else 'cpu'

print(opt)