
# Time-series (taking Goods as demo)
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
#warnings.filterwarning("ignore")
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

df = pd.read_csv('Data/Goods.csv')

df['StartDate'].min(), df['StartDate'].max()
df['EndDate'].min(), df['EndDate'].max()

df.columns
cols = ['RequestID', 'EndDate', 'AgencyName',
       'CategoryDescription', 'SelectionMethodDescription']


# change the format of time and set the DatetimeIndex
from datetime import datetime
time_df = df.drop(cols, axis=1)
time = list(time_df['StartDate'])
time = [datetime.strptime(time,'%m/%d/%Y').strftime('%Y-%m-%d') for time in time]
time_df['StartDate'] = time
time_df = time_df.sort_values('StartDate')
time_df.isnull().sum()
time_df = time_df.groupby('StartDate')
timeseries = time_df['ContractAmount'].sum().reset_index()

timeseries = timeseries.set_index(pd.DatetimeIndex(timeseries['StartDate']))
timeseries.index

# Taking the sum of each month/quarter

monthly = timeseries['ContractAmount'].resample('MS').sum()

quarterly = timeseries['ContractAmount'].resample('Q').sum()
timeseries.index

monthly['2020':]
quarterly['2020':]

# Visulaizing

monthly.plot(figsize=(15,6))
plt.show()
quarterly.plot(figsize=(15,6))
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 18,8

decomposition = sm.tsa.seasonal_decompose(monthly, model='additive')
fig = decomposition.plot()
plt.show()

# SARIMA to time series forecasting
#he models notation is SARIMA(p, d, q).(P,D,Q)m
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter for SARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

AIC = []
r_df = pd.DataFrame(columns = ['Param','Seasonal','AIC'])

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(monthly, order=param, seasonal_order=param_seasonal,
                                            enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            r_df = r_df.append({'Param': param, 'Seasonal': param_seasonal, 'AIC': results.aic}, ignore_index=True)
            #print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
        except:
            continue

#Find the min value of AIC (optimal option)
r_df.iloc[r_df['AIC'].idxmin()]

mod = sm.tsa.statespace.SARIMAX(monthly,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(18, 8))
plt.show()

##Predition
from pandas.plotting import register_matplotlib_converters

pred = results.get_prediction(start=pd.to_datetime('2016-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = monthly['2010':].plot(label='observed')

#plt.plot

pred.predicted_mean.plot()


ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Contract Amount')
plt.legend()
plt.show()
