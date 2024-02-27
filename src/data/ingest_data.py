"""
Large Language Model College Football Data: Ingest Raw Data
Author: Trevor Cross
Last Updated: 02/27/24

Initial ETL of data from collegefootballdata.com. Extract available data,
fix column issues, remove unwanted columns, and save locally as CSV.
"""

#%% ----------------------
#-- ---Import Libraries---
#-- ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import data saving library
from mlem.api import save

# import support functions
import os
import sys
import json
from tqdm import tqdm

# import toolbox
sys.path.append("src")
from toolbox import *

#%% ----------------------------
#-- ---Obtain ETL Credentials---
#-- ----------------------------

# get API key
key_path = os.path.join('..', '..', 'secrets', 'cfd_api_key.json')
with open(key_path) as json_file:
    api_key = json.load(json_file)['api_key']

#%% -----------------------
#-- ---Extract Game Data---
#-- -----------------------

# define base URL to 'games' section
base_url = 'https://api.collegefootballdata.com/games'

# define section filters (year & season_type)
cart_prod = [(year, season_type) for year in np.arange(2001, 2025)
                                 for season_type in ['regular', 'postseason']]

# define list of URLs
urls = [f'{base_url}?year={year}&seasonType={season_type}' for year, season_type in cart_prod]

# make requests & wrap in df
resp_df = pd.concat( [pd.DataFrame(make_request(url, api_key)) for url in tqdm(urls, desc="Making 'game' requests")] )

# save resp_df using MLEM
output_path = os.path.join('data', 'raw', 'game_data.csv')
save(resp_df, output_path)

#%% ---------------------------
#-- ---Extract FBS Team Data---
#-- ---------------------------

# define base URL to 'teams' section
base_url = 'https://api.collegefootballdata.com/teams/fbs'

# make request
print("Making FBS Team Data request...")
resp_df = pd.DataFrame(make_request(base_url, api_key))
resp_df.drop(columns=['logos','location'], inplace=True)
print("...Finished.")

# save resp_df using MLEM
output_path = os.path.join('data', 'raw', 'team_data.csv')
save(resp_df, output_path)

#%% -----------------------------
#-- ---Extract Recruiting Data---
#-- -----------------------------

# define base URL to 'recruiting/teams' subsection
base_url = 'https://api.collegefootballdata.com/recruiting/teams'

# build list of URLs
urls = [f'{base_url}?year={year}' for year in np.arange(2001, 2025)]

# make requests & append responses
resp_df = pd.concat( [pd.DataFrame(make_request(url, api_key)) for url in tqdm(urls, desc="Making 'recruiting' requests")] )

# save resp_df using MLEM
output_path = os.path.join('data', 'raw', 'recr_data.csv')
save(resp_df, output_path)
