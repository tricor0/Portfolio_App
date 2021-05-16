import requests
import json

API_KEY = "PK9PY0D4HKTZK8HFA61G"
SECRET_KEY = "xA27qPo7GEHFIJxhGWamstheu2R87AbOs4lL52N6"

BASE_URL = "https://paper-api.alpaca.markets"
ACCOUNT_URL = "{}/v2/account".format(BASE_URL)
ORDER_URL = "{}/v2/orders".format(BASE_URL)
HEADERS = {'APCA-API-KEY-ID': API_KEY, 'APCA-API-SECRET-KEY': SECRET_KEY}

def get_account():
    r = requests.get(ACCOUNT_URL, headers=HEADERS)
    return json.loads(r.content)
def get_buying_power():
    r = requests.get(ACCOUNT_URL, headers=HEADERS)
    content = json.loads(r.content)
    return content.get('buying_power')


def create_order(symbol, qty, side, type, time_in_force):
    data = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": type,
        "time_in_force": time_in_force
    }

    r = requests.post(ORDER_URL, json=data, headers=HEADERS)

    return json.loads(r.content)

def get_orders():
    r = requests.get(ORDER_URL, headers=HEADERS)

    return json.loads(r.content)

# response = create_order('AAPL', 100, "buy", "market", "gtc")
# response = create_order('MSFT', 1000, "buy", "market", "gtc")
# orders = get_orders()
account = get_account()
print(account)
