# Description:  This program uses an artificial recurrent nearal network called Long Sort Term Memory (LSTM)
#               to predict the closing stock price of a corporation (Apple in this case) using the past 60 day stock price.
#               https://www.youtube.com/watch?v=QIUxPv5PJOY&ab_channel=ComputerScience
#Import the libraries
import math
#pip3.8 install pandas_datareader
# import pandas_datareader as web
import numpy as np
from matplotlib import pyplot as plt

import data_agent
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
# import matplotlib.pyplot as plt
# plt.style.use('fivethirtyeight')
#Import the libraries for gui
# from tkinter import *
#Create window for gui
# root = Tk()
# root.title("Address Book Program")
# root.iconbitmap('C:\PythonPrograms\Tkinter Course\money.ico')
# root.geometry("400x400")
def calculate_LSTM(company):
#Get the stock quote
    data_agent.get_stock_data(company)
    # df = web.DataReader('LGD1L.VS', data_source='yahoo', start='2012-01-01', end='2021-05-15')
    #Show the data
    #print(df)
    #Get the number of rows and columns in the data set

    # def closePriceHistory():
    #     #Visualize the closing price history
    #     plt.figure(figsize=(16,8))
    #     plt.title('Close Price History')
    #     plt.plot(df['Close'])
    #     plt.xlabel('Date', fontsize=18)
    #     plt.ylabel('Close Price USD ($)', fontsize=18)
    #     plt.show()

    #Company name and close price graph
    # company_name_label = Label(root, text="Selected Company Name: LGD1L.VS")
    # company_name_label.pack()
    # graph_button = Button(root, text="Show Close Price History Graph", command=closePriceHistory)
    # graph_button.pack()


    #Create a new dataframe with only the close column
    data = data_agent.get_df().filter(['Close'])
    #Convert the dataframe to a numpy array
    dataset = data.values
    #Get the number of rows to train the model on
    training_data_len = math.ceil(len(dataset)* .8 )

    #training_data_len

    #Scale the data
    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(dataset)

    #scaled_data

    #Create the training data set
    #Create the scaled trainig data set
    train_data = scaled_data[0:training_data_len , :]
    #Split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    print(x_train)

    for i in range(60, len(train_data)):
      x_train.append(train_data[i-60:i,0])
      y_train.append(train_data[i, 0])
    """
      #See how train datasets are created
      if i<=62:
        print(x_train)
        print(y_train)
        print()
    """

    #Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    #Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

    #Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences = False))
    model.add(Dense(25))
    model.add(Dense(1))
    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')


    #Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=5)
    # trainModel()
    # train_button = Button(root, text="Train Model", command=trainModel)
    # train_button.pack()


    #Create the testing data set
    #Create a new array containing scaled values from integers 1543 to 2003
    test_data = scaled_data[training_data_len - 60: , :]
    #Create the data sets x_test and y_test
    x_test=[]
    y_test = dataset[training_data_len:, :]
    for i in range (60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
  
    #Convert the data to numpy array
    x_test = np.array(x_test)

    #Reshape the data from 2D to 3D
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    #Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    #Get the root mean squared error (RMSE)
    #Value of 0 (almost never achieved in practice) would indicate a perfect fit to the data
    rmse = np.sqrt(np.mean(predictions - y_test )**2)
    print("Model RMSE: " + str(rmse))

    # def modelPredictionGraph():

    #Plot the data
    #train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    #Visualize the data
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    #plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    #plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.legend(['Val', 'Predictions'], loc='lower right')
    plt.show()

# model_button = Button(root, text="Real Value vs Predictions Graph", command=modelPredictionGraph)
# model_button.pack()

#Show the valid and predicted prices
#valid

# TODO pritaikyti situos metodus. kazkur issaugoti modeli, kad nereiketu jo kurti kiekviena karta.
def predictOneDay():
    #Get the quote
    apple_quote = web.DataReader('GOOGL', data_source='yahoo', start='2012-01-01', end='2021-04-15') ## end='2019-12-17'
    #Create a new dataframe
    new_df = apple_quote.filter(['Close'])
    #Get the last 60 day closin price values and convert the dataframe to an array
    last_60_days = new_df[-60:].values
    #Scale the data
    last_60_days_scaled= scaler.transform(last_60_days)
    #Create an empty list
    X_test=[]
    #Append the past 60 days
    X_test.append(last_60_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape
    X_test= np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1 ))
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #Undo the scalling
    pred_price = scaler.inverse_transform(pred_price)
    #print("Predicted Price for 2021-04-16: " + str(pred_price[0][0]))
    predictionForTomorrow = "Predicted Price for 2021-04-16: " + str(pred_price[0][0])
    
    predictOneDay_label = Label(root, text=predictionForTomorrow)
    predictOneDay_label.pack()


# predictoneday_button = Button(root, text="Predict One Day", command=predictOneDay)
# predictoneday_button.pack()

def getRealValue():
    #Get the quote
    apple_quote2 = web.DataReader('GOOGL', data_source='yahoo', start='2021-04-16', end='2021-04-16') ##atart and end='2019-12-18'
    #print("Real Price for 2021-04-16: " + str(apple_quote2['Close'][1]))
    realValueForTomorrow = "Real Price for 2021-04-16: " + str(apple_quote2['Close'][1])
    
    realValueForOneDay_label = Label(root, text=realValueForTomorrow)
    realValueForOneDay_label.pack()

# getRealValue_button = Button(root, text="Get Real Value for One Day", command=getRealValue)
# getRealValue_button.pack()

#Run main window
# root.mainloop()
