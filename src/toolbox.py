"""
Large Language Model College Football Data: Toolbox
Author: Trevor Cross
Last Updated: 03/25/24

Series of functions used to assist in manipulating data from collegefootballdata.com.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import support libraries
import requests as req
import json

# --------------------------
# ---Define ETL Functions---
# --------------------------

# define function to make requests
def make_request(url, api_key):

    # define headers
    headers = {'Content-Type': 'application/json',
               'Authorization': f'Bearer {api_key}'}

    # return JSON response
    return req.get(url, headers=headers).json()

# ------------------------------
# ---Define Support Functions---
# ------------------------------

# define function to load local JSON file as Python dict
def json_to_dict(file_path):
    with open(file_path) as file:
        return json.load(file)
