import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import app


def get_files():
    directory = f'{app.__ROOT__}/Data/'

    # load all the files in the directory
    return [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]

# Load data
data_files = get_files()
data_list = [pd.read_csv(file, parse_dates=['time'], index_col='time') for file in data_files]
data = pd.concat(data_list)

# Preprocess data (example: resampling to hourly data)
data = data.resample('s').mean().ffill()
# Fit ARIMA model
model = ARIMA(data['main'], order=(5,1,0))
model_fit = model.fit()

# Generate synthetic data
synthetic_data = model_fit.forecast(steps=1000000)  # Adjust steps as needed
synthetic_data.index = pd.date_range(start=data.index[-1], periods=1000000, freq='s')

# Plot original and synthetic data
plt.figure(figsize=(12, 6))
plt.plot(data['main'], label='Original Data')
plt.plot(synthetic_data, label='Synthetic Data', linestyle='--')
plt.legend()
plt.show()

# Save synthetic data
synthetic_data.to_csv('synthetic_power_usage.csv')