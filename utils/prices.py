import requests
import json
import os

def fetch_and_save_coin_list(filename="coin_list.json"):
    """
    Fetches the coin list from the API and saves it to a local JSON file.
    """
    url = "https://api.coingecko.com/api/v3/coins/list"
    headers = {"accept": "application/json"}
    
    # Make the API request
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # Save to file
        with open(filename, "w") as file:
            json.dump(response.json(), file)
        print(f"Coin list successfully saved to {filename}.")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")

def load_coin_list(filename="coin_list.json"):
    """
    Loads the coin list from a local JSON file. If the file does not exist,
    fetches the data from the API and saves it.
    """
    if os.path.exists(filename):
        # Load data from the file
        with open(filename, "r") as file:
            coin_list = json.load(file)
        print("Coin list loaded from local file.")
    else:
        # Fetch data from API and save
        print(f"{filename} not found. Fetching from API...")
        fetch_and_save_coin_list(filename)
        with open(filename, "r") as file:
            coin_list = json.load(file)
    return coin_list

def get_coin_id_by_name(name, coin_list):
    """
    Finds the coin ID based on its name from the provided coin list.
    """
    for coin in coin_list:
        if coin['name'].lower() == name.lower():
            return coin['id']
    return None

# Function to fetch cryptocurrency price data
def fetch_crypto_price(coin_id):
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": coin_id,  # The coin ID
        "vs_currencies": "usd",
        "include_24hr_change": "true"
    }
    response = requests.get(url, params=params)
    data = response.json()
    current_price, change_1d = None, None
    for key, value in data.items():
        current_price = round(value.get('usd'), 3)
        change_1d = round(value.get('usd_24h_change'), 5) 
    return current_price, change_1d

