# Stock Price Prediction with Deep Learning

Implement data_cleaning.py to get .csv data

Find out your favorite stock in Yahoo Finance Page, and find out the ID of that Finance

https://finance.yahoo.com/quote/AAPL?p=AAPL

<hr/>

## Environment

- CUDA 10.1
- pytorch 1.3.1
- python 3.6.x

<hr/>

## Implement

#### **To Download Apple's prices**
```bash
cd data
python3 data_cleaning.py AAPL
```

#### **Implement visdom (localhost:8097)**
```
python3 -m visdom.server
or
visdom
```

#### **To train Apple's price by an LSTM model and save the model**
```
python3 main.py --mode train --dataset AAPL --model LSTM
```

#### **To test the performance of the model**
```
python3 main.py --mode test --dataset AAPL --model LSTM
```
