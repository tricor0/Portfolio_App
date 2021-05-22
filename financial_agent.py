import requests
import json


API_KEY = "PK6LCE46YT6P1T39GRXB"
SECRET_KEY = "odhWhU3BXCsistPIueFsRl7vgkRvNRlKxez7QjwV"

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

account = get_account()
print(account)
