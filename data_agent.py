import pandas_datareader as web
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def get_stock_data(company):
    global df
    df = web.DataReader(company, data_source='yahoo', start='2012-01-01', end='2021-05-15')
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
