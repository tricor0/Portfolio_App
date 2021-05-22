import pandas_datareader as web
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

global start_date
start_date = '2012-01-01'
global end_date
end_date = datetime.today().strftime('%Y-%m-%d')

def get_stock_data(company):
    global df
    df = web.DataReader(company, data_source='yahoo', start=start_date, end=end_date)
    df.shape

def close_price_history(company):
    #Visualize the closing price history
    plt.figure(figsize=(8,4))
    plt.title('Uždarymo kainų istorija ' + company)
    plt.plot(df['Close'])
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Uždarymo kainų istorija ($)', fontsize=18)
    plt.show()

def get_df():
    return df

def load_saved_agent(path):
    return keras.models.load_model(path)
