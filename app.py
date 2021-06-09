from flask import Flask,render_template,request,redirect
import tweepy
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
from PIL import Image
import PIL
import numpy as np
from django.http import HttpResponse
import investpy
from datetime import datetime, timedelta
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
import investpy
import plotly.graph_objects as go
from keras.preprocessing.sequence import TimeseriesGenerator

app = Flask(__name__)

app.vars={'stock name'}
x= 'dummy'

def data_splitting(Data, training_percentage, val_percentage, feature):
    sub_data = Data[feature].values
    sub_data = sub_data.reshape((-1,1))

    split1 = int(training_percentage*len(sub_data))
    split2 = int((training_percentage+val_percentage)*len(sub_data))

    sub_data_train = sub_data[:split1]
    sub_data_val = sub_data[split1:split2]
    sub_data_test = sub_data[split2:]

    date_train = Data['Date'][:split1]
    date_val = Data['Date'][split1:split2]
    date_test = Data['Date'][split2:]

    return date_train, sub_data_train, date_val, sub_data_val, date_test, sub_data_test


def create_model(window_size):
    model = Sequential()
    model.add(LSTM(units=128,activation='tanh',return_sequences=True,input_shape=(window_size ,1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64,activation='tanh',return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32,activation='tanh',return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=16,activation='linear'))
    model.add(Dense(units=1,activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    #model.summary()
    return model

def generator(open_train, open_val, open_test, close_train, close_val, close_test, window_size, batch_size):
    train_generator = TimeseriesGenerator(close_train, close_train, length=window_size, batch_size=batch_size)  
    val_generator = TimeseriesGenerator(close_val, close_val, length=window_size, batch_size=batch_size)     
    test_generator = TimeseriesGenerator(close_test, close_test, length=window_size, batch_size=batch_size)
    return train_generator, val_generator, test_generator

def predict(stock, num_prediction, model,look_back):
    close_data=stock['Close'].values
    close_data = close_data.reshape((-1))
    prediction_list = close_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
    
def predict_dates(stock, num_prediction):
    last_date = stock['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

def make_pred():
    today = datetime.today().strftime('%d/%m/%Y')
    start_date = (datetime.today() - timedelta(days=2001)).strftime('%d/%m/%Y')

    df = investpy.get_stock_historical_data(stock='2010', country='saudi arabia', from_date=start_date, to_date= today)
    df = df.drop(columns= ['Currency', 'Volume'])
    sabic= df.copy()
    
    
    sabic.reset_index(level=0, inplace=True)
    sabic.drop(columns=['Open','High', 'Low'], inplace=True)

    date_train, close_train, date_val, close_val, date_test, close_test = data_splitting(sabic, 0.7, 0.15, 'Close')
    window_size = 10
    batch_size = 32
    num_epochs = 10000

    model = create_model(window_size)
    early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', min_delta=0, patience=20, verbose=0,
                        mode='auto', baseline=None, restore_best_weights=True)

    train_generator, val_generator, test_generator = generator(close_train, close_val, close_test, close_train, close_val, close_test, window_size, batch_size)
    history = model.fit(train_generator, epochs=num_epochs, verbose=0, validation_data=val_generator, callbacks=early_stopping_monitor)
    test_mse= model.evaluate(test_generator, verbose=0)
    
    num_prediction = 5
    forecast = predict(sabic, num_prediction, model,window_size)
    forecast_dates = predict_dates(sabic, num_prediction)
    
    result = pd.DataFrame()
    result['Date'] = forecast_dates
    result['close'] = forecast
    return result, test_mse

result, test_mse = make_pred()

@app.route('/')
def Main():
    global result, test_mse
    return render_template('Output.html', tables=[result.to_html()], titles=result.columns.values, value=test_mse)

if __name__ == "__main__":
    app.run(debug=True)