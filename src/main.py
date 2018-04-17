# Load libraries
import numpy as np
import pandas
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import quandl
import os

quandl.ApiConfig.api_key = os.environ["API_CREDENTIALS"]

#load data
data = quandl.get('NSE/OIL')
print("dataset", data.shape)

