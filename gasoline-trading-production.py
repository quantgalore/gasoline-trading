# -*- coding: utf-8 -*-
"""
Created in 2023

@author: Quant Galore
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from feature_functions import Binarizer, return_proba
from pandas_market_calendars import get_calendar

start_time = datetime.now()

fmp_api_key = "0f1bb59009e6ef4747289a60586bde4f"
polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
calendar = get_calendar("NYSE")

start_date = "2018-01-01"
end_date = (datetime.today()).strftime("%Y-%m-%d")

available_commodities = pd.json_normalize(requests.get(f"https://financialmodelingprep.com/api/v3/symbol/available-commodities?apikey={fmp_api_key}").json())
available_commodities["short_symbol"] = available_commodities["symbol"].apply(lambda x: x.split('U')[0])

cot_by_dates = pd.json_normalize(requests.get(f"https://financialmodelingprep.com/api/v4/commitment_of_traders_report_analysis?from={start_date}&to={end_date}&apikey={fmp_api_key}").json()).set_index("date")
cot_by_dates.index = pd.to_datetime(cot_by_dates.index)

ticker = "RB"

ticker_cot = cot_by_dates[cot_by_dates["symbol"] == ticker].copy().sort_values(by="date", ascending = True)
ticker_cot["report_date"] = ticker_cot.index + timedelta(days = 3)

full_ticker = available_commodities[available_commodities["short_symbol"] == ticker]["symbol"].iloc[0]

ticker_performance = pd.json_normalize(requests.get(f"https://financialmodelingprep.com/api/v3/historical-price-full/{full_ticker}?apikey={fmp_api_key}").json()["historical"]).set_index("date")
ticker_performance.index = pd.to_datetime(ticker_performance.index)

trade_dates = pd.DataFrame({"trade_dates": calendar.schedule(start_date = "2023-01-01", end_date = end_date).index}).set_index("trade_dates")

features = ['currentLongMarketSituation',
       'currentShortMarketSituation', 'netPostion',
       'changeInNetPosition', 'marketSentiment']

target = "pct_change"

trades = []
days = 1

# assumes you run this on Friday-Sunday
date = trade_dates.index[-1] - timedelta(days = 3)

cot_of_date = ticker_cot[ticker_cot.index == date].copy()
cot_of_date['marketSentiment'] = cot_of_date['marketSentiment'].str.strip()

if len(cot_of_date) < 1:
    raise Exception

report_date = date + timedelta(days = 3)

ticker_returns = ticker_performance.copy().reset_index().sort_values(by="date", ascending = True).set_index("date")
# get the next session's return
ticker_returns["pct_change"] = ((ticker_returns["adjClose"].shift(-days) - ticker_returns["adjClose"]) / ticker_returns["adjClose"]) * 100
# the return over the weekend following the release (monday close)
ticker_returns = ticker_returns[ticker_returns.index.isin((ticker_cot["report_date"] + timedelta(days = 3)).values)]

#

# get the prior cot's
cot_for_ticker = ticker_cot.copy()
cot_for_ticker["report_date"] = cot_for_ticker.index + timedelta(days = 3)
cot_for_ticker["monday_date"] = cot_for_ticker["report_date"] + timedelta(days=3)
cot_for_ticker = cot_for_ticker.set_index("monday_date")

# line up the cot report with what the future did over the weekend
merged = pd.concat([cot_for_ticker, ticker_returns], axis = 1).dropna()
merged['marketSentiment'] = merged['marketSentiment'].str.strip()

# isolate only the cot's before this report
historical_cot = merged[merged.index < report_date].copy()

transformed_features = pd.get_dummies(historical_cot[features].copy())

X = transformed_features
Y = historical_cot[target].apply(Binarizer).values

RandomForest_Model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None).fit(X,Y)

# process the current cot
pre_prod_data = pd.get_dummies(pd.concat([historical_cot, cot_of_date], axis = 0)[features])
prod_data = pre_prod_data.tail(1).reset_index(drop=True).copy()

random_forest_prediction = RandomForest_Model.predict(prod_data)
random_forest_prediction_probability = RandomForest_Model.predict_proba(prod_data)

random_forest_prediction_dataframe = pd.DataFrame({"prediction": random_forest_prediction})
random_forest_prediction_dataframe["probability_0"] = random_forest_prediction_probability[:,0]
random_forest_prediction_dataframe["probability_1"] = random_forest_prediction_probability[:,1]
random_forest_prediction_dataframe["probability"] = return_proba(random_forest_prediction_dataframe)

prediction = random_forest_prediction[0]
probability = random_forest_prediction_dataframe["probability"].iloc[0]

if prediction == 1:
    trade_string = f"{ticker} is expected to go up over the weekend with a {round(probability*100,2)}% probability as of {report_date.strftime('%Y-%m-%d')}"
elif prediction == 0:
    trade_string = f"{ticker} is expected to go down over the weekend with a {round(probability*100,2)}% probability as of {report_date.strftime('%Y-%m-%d')}"
    
print(trade_string)
