import financial_agent
import data_agent
from datetime import datetime
import math


# today = str(datetime.today().strftime('%Y-%m-%d'))
# today = '2020-04-29'
def execute_LSTM_strategy(company, today_price, one_day_prediction):
    response = ""
    if (today_price < one_day_prediction):
        print("I buy for " + str(today_price))
        response = financial_agent.create_order(company, 10, "buy", "market", "gtc")
    if (today_price > one_day_prediction):
        print("I sell for " + str(today_price))
        response = financial_agent.create_order(company, 10, "sell", "market", "gtc")
    print(response)

def execute_fibonacci_retracement(company, dataframe):
    response = ""
    # print(dataframe)
    # for col in dataframe.columns:
    #     print(col)
    # reduced_df = dataframe[["Date", "Buy_Signal_Price", "Sell_Signal_Price"]]
    # calc_for_today = reduced_df.item([(reduced_df['Date'] == today)])
    #reduced_df.loc[reduced_df['Date'] == today]
    # result = reduced_df[(reduced_df['Date']== today)]
    # print(reduced_df.loc[reduced_df['Date'] == today])
    # print(type('2021-05-05'))
    # print("Date compared to today date: " + str((reduced_df.loc[reduced_df['Date'] == today])))
    # print(dataframe.at[today,'Date'])
    # if (calc_for_today):
    # #     print("Is today buy signal nan: " + (reduced_df.loc[math.isnan(float(reduced_df['Buy_Signal_Price']))]))

    # buy_signal = reduced_df.loc[float(reduced_df['Buy_Signal_Price'])]
    buy_signal = dataframe.at[data_agent.get_end_date(),'Buy_Signal_Price']
    # sell_signal = reduced_df.loc[float(reduced_df['Sell_Signal_Price'])]
    sell_signal = dataframe.at[data_agent.get_end_date(),'Sell_Signal_Price']
    if not math.isnan(buy_signal):
        print("I buy for " + str(buy_signal))
        response = financial_agent.create_order(company, 10, "buy", "market", "gtc")
    if not math.isnan(sell_signal):
        print("I sell for " + str(sell_signal))
        response = financial_agent.create_order(company, 10, "sell", "market", "gtc")
    else:
        print("I dont do anything")
        return
    print(response)
