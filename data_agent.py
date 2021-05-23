import pandas_datareader as web
import pandas as pd
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

global start_date
start_date = '2012-01-01'
global end_date
end_date = datetime.today().strftime('%Y-%m-%d')


def get_stock_data(company):
    return web.DataReader(company, data_source='yahoo', start=start_date, end=end_date)

def get_stock_data_Fib(company):
    #Get and show the data
    # df = pd.read_csv('GOOG.csv')
    dataframe = web.DataReader(company, data_source='yahoo', start=start_date, end=end_date)
    #Set the date as the index
    # df = df.set_index(web.DatetimeIndex(df['Date'].values))
    dataframe['Date'] = dataframe.index
    dataframe = dataframe.set_index(pd.DatetimeIndex(dataframe['Date'].values))
    #Show the data
    return dataframe

def get_stock_data_LSTM(company):
    global df
    df = get_stock_data(company)
    df.shape


def close_price_history(company):
    #Visualize the closing price history
    plt.figure(figsize=(11,4))
    plt.title('Uždarymo kainų istorija ' + company)
    plt.plot(df['Close'])
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Vertė ($USD)', fontsize=18)
    plt.show()

def get_df():
    return df

def load_saved_agent(path):
    return keras.models.load_model(path)
