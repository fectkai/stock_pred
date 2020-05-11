import argparse
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--type', type=str, help='price, log, volat', default='price')
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--dataset', type=str, default='KOSPI')
parser.add_argument('--model', type=str, default='LSTM')
parser.add_argument('--bs', dest='batch_size', help='batch size', default=8, type=int)
parser.add_argument('--epoch', dest='num_epochs', default=2000, type=int)
parser.add_argument('--debug_mode', dest='debug_mode', action='store_true')
parser.set_defaults(debug_mode=False)

parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)

parser.add_argument('--bin', dest='binary', action='store_true')
parser.set_defaults(binary=False)

opt = parser.parse_args()

opt.device='cuda' if torch.cuda.is_available() else 'cpu'

print(opt)