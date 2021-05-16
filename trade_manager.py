import financial_agent

def execute_LSTM_strategy(company, today_price, one_day_prediction):
    response = ""
    if (today_price < one_day_prediction):
        response = financial_agent.create_order(company, 10, "buy", "market", "gtc")
    if (today_price > one_day_prediction):
        response = financial_agent.create_order(company, 10, "sell", "market", "gtc")
    print(response)
