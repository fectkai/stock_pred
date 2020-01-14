# Stock Price Prediction with Deep Learning

implement data_cleaning.py to get .csv data

- To Download Apple's prices
```bash
python3 data_cleaning.py AAPL
```

- To train Apple's price by an LSTM model and save the model
```
python3 main.py train AAPL LSTM
```
- To test the performance of the model
```
python3 main.py test AAPL LSTM
```
