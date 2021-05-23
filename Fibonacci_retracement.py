# -*- coding: utf-8 -*-
"""Fibonacci Retracement v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11tE3WpqfqUsQVikjSg4j-Vw8IH5LdEpG
"""

#Description: This program uses Fibonacci Retracement Levels and MACD to indicate when to buy and sell stock.

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_agent
plt.style.use('fivethirtyeight')


#Get and show the data
# df = pd.read_csv('GOOG.csv')
# df = web.DataReader('GOOG', data_source='yahoo', start='2020-04-20', end='2021-04-16')
#Set the date as the index
# df = df.set_index(web.DatetimeIndex(df['Date'].values))
# df['Date'] = df.index
# df = df.set_index(pd.DatetimeIndex(df['Date'].values))
#Show the data
# df

def calculate_and_get_dataframe(company):
    dataframe = prepare_calculations(company)
    get_dataframe_with_signals(dataframe)
    plot_with_signals(dataframe)

def prepare_calculations(company):
    dataframe = data_agent.get_stock_data_Fib(company)

    #plot the data
    # plt.figure(figsize=(12,4))
    # plt.plot(dataframe.Close)
    # plt.title('Uždarymo kaina')
    # plt.xlabel('Data')
    # plt.ylabel('Vertė ($USD)')
    # plt.xticks(rotation=45)
    # plt.show()

    #Define for later
    global max_price
    global min_price
    global difference
    global first_level
    global second_level
    global third_level
    global fourth_level

    #Calculate the Fibonacci Retracement Levels
    max_price = dataframe['Close'].max()
    min_price = dataframe['Close'].min()

    difference = max_price - min_price
    first_level = max_price - difference * 0.236
    second_level = max_price - difference * 0.382
    third_level = max_price - difference * 0.5
    fourth_level = max_price - difference * 0.618

    #Calculate the MACD line and the signal line indicators
    #Calculate the Short Term Exponantial Moving Average
    ShortEMA = dataframe.Close.ewm(span=12, adjust=False).mean()
    #Calculate the Long Term Exponential Moving Average
    LongEMA = dataframe.Close.ewm(span=26, adjust=False).mean()
    #Calculate the Moving Average Convergence/Divergernce (MACD)
    MACD = ShortEMA - LongEMA
    #Calculate the Signal Line
    signal = MACD.ewm(span=9, adjust=False).mean()

    #Plot the Fibonacci Levels along with the close price and the MACD and Signal Line
    new_df = dataframe

    #plot the Fibonacci Levels

    # plt.figure(figsize=(12.33, 9.5))
    # plt.subplot(2,1,1)
    # plt.plot(new_df.index, new_df['Close'])
    # plt.axhline(max_price, linestyle='--', alpha=0.5, color='red')
    # plt.axhline(first_level, linestyle='--', alpha=0.5, color='orange')
    # plt.axhline(second_level, linestyle='--', alpha=0.5, color='yellow')
    # plt.axhline(third_level, linestyle='--', alpha=0.5, color='green')
    # plt.axhline(fourth_level, linestyle='--', alpha=0.5, color='blue')
    # plt.axhline(min_price, linestyle='--', alpha=0.5, color='purple')
    # plt.ylabel('Fibonacci')
    # frame1=plt.gca()
    # frame1.axes.get_xaxis().set_visible(False)

    #Plot the MACD Line and the Signal Line
    # plt.subplot(2,1,2)
    # plt.plot(new_df.index, MACD)
    # plt.plot(new_df.index, signal)
    # plt.ylabel('MACD')
    # plt.xticks(rotation=45)
    #
    # plt.savefig('fig1.png')

    #Create new columns for the data frame
    dataframe['MACD'] = MACD
    dataframe['Signal Line'] = signal
    #show the new data
    # df
    return dataframe

#Create a function to be used in our strategy to get upper Fibonacci level and the Lower Fibonacci level
def getLevels(price):
    if price >= first_level:
        return (max_price, first_level)
    elif price >= second_level:
        return (first_level, second_level)
    elif price >= third_level:
        return (second_level, third_level)
    elif price >= fourth_level:
        return (third_level, fourth_level)
    else:
        return (fourth_level, min_price)

#Create a function for the trading strategy

#The strategy
#If the signal line crosses above the MACD line and the current price crossed above or below the last Fibonacci level then buy
#If the signal line crosses below the MACD line and the current price crossed above or below the last Fibonacci level then sell
#Never sell at a price that's lower than I bought

def strategy(dataframe):
    buy_list= []
    sell_list=[]
    flag= 0
    last_buy_price =0

    #Loop through the data set
    for i in range(0, dataframe.shape[0]):
      price =dataframe['Close'][i]
      #If this is the first data point within the data set, then get the level above and below it.
      if i == 0:
        upper_lvl, lower_lvl = getLevels(price)
        buy_list.append(np.nan)
        sell_list.append(np.nan)
      #Else if the current price is greater than or equal to the upper_lvl, or less than or equal to the lower_lvl, then we know the price has 'hit' or crossed the new Fibonacci level
      elif price >= upper_lvl or price <= lower_lvl:
        #Check to see if the MACD line crossed above or below the signal line
        if dataframe['Signal Line'][i] > dataframe['MACD'][i] and flag == 0:
          last_buy_price=price
          buy_list.append(price)
          sell_list.append(np.nan)
          #Set the flag to 1 to signal that the share was bought
          flag = 1
        elif dataframe['Signal Line'][i] < dataframe['MACD'][i] and flag == 1 and price > last_buy_price:
          buy_list.append(np.nan)
          sell_list.append(price)
          #Set the flag to 0 to signal that the share was sold
          flag = 0
        else:
          buy_list.append(np.nan)
          sell_list.append(np.nan)
      else:
          buy_list.append(np.nan)
          sell_list.append(np.nan)

      #Update the new levels
      upper_lvl, lower_lvl =getLevels(price)

    return buy_list, sell_list

def get_dataframe_with_signals(dataframe):
    #Create buy and sell columns
    buy, sell = strategy(dataframe)
    dataframe['Buy_Signal_Price'] = buy
    dataframe['Sell_Signal_Price'] = sell
    #show the data
    return dataframe

def plot_with_signals(dataframe):
    #Plot the Fibonacci Levels along with the close price and with the Buy and Sell signals
    # new_df = df
    #plot the Fibonacci Levels
    plt.figure(figsize=(20.33, 4.5))
    plt.plot(dataframe.index, dataframe['Close'], alpha=0.5)
    plt.scatter(dataframe.index, dataframe['Buy_Signal_Price'], color='green', marker='^', alpha=1)
    plt.scatter(dataframe.index, dataframe['Sell_Signal_Price'], color='red', marker='v', alpha=1)
    plt.axhline(max_price, linestyle='--', alpha=0.5, color='red')
    plt.axhline(first_level, linestyle='--', alpha=0.5, color='orange')
    plt.axhline(second_level, linestyle='--', alpha=0.5, color='yellow')
    plt.axhline(third_level, linestyle='--', alpha=0.5, color='green')
    plt.axhline(fourth_level, linestyle='--', alpha=0.5, color='blue')
    plt.axhline(min_price, linestyle='--', alpha=0.5, color='purple')
    plt.ylabel('Close Price in USD')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.show()