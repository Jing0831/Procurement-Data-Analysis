import pandas as pd

df = pd.read_csv('April 17, 2021.csv')
df.info()
df.columns

df_raw = df[['RequestID', 'StartDate', 'EndDate', 'AgencyName',
       'CategoryDescription',
       'SelectionMethodDescription', 'ContractAmount']]


# Function for searching the short notation by using RequestID number:
df_shorttitle = df[['RequestID','ShortTitle']]
def notation(ID_num):
    filt = df_shorttitle['RequestID'] == ID_num
    print(df_shorttitle.loc[filt])

notation(20071004002)

# Clean dataset

df_raw.isnull().sum()
df_raw.shape
df_clean = df_raw.dropna(how = 'any')
df_clean = df_clean[df_clean['ContractAmount'] > 0]
df_clean.info()
df_clean.columns
df_clean.to_csv('Data/cleandataset.csv')
df_clean = pd.read_csv('Data/cleandataset.csv')

# Bids counted by category:
CategoryDescription = df_clean['CategoryDescription'].value_counts()
categorysummary_df = pd.DataFrame({'Counts of Bids':CategoryDescription})
categorysummary_df.reset_index(inplace=True)
categorysummary_df.rename(columns={'index':'Category'},inplace=True)
categorysummary_df.to_csv('Data/CategorySummary.csv',index = False)

# group by Category Description

category_group = df_clean.groupby(['CategoryDescription'])
Goods = category_group.get_group('Goods')
Goods.value_counts()
Goods.to_csv('Data/Goods.csv', index = False)
HumanClientServices = category_group.get_group('Human Services/Client Services')

OtherServices = category_group.get_group('Services (other than human services)')
Construction = category_group.get_group('Construction/Construction Services')
GoodsServices = category_group.get_group('Goods and Services')
ConstructionServices = category_group.get_group('Construction Related Services')

# Analyze by category:
# Selection Methods for each category
# BM_goods_df
BM = Goods['SelectionMethodDescription'].value_counts(normalize=True)
BM_goods_df = pd.DataFrame({'Goods':BM})

# BM_HumanClientServices_df
BM = HumanClientServices['SelectionMethodDescription'].value_counts(normalize=True)
BM_HumanClientServices_df = pd.DataFrame({'HumanClientServices':BM})

# BM_OtherServices_df
BM = OtherServices['SelectionMethodDescription'].value_counts(normalize=True)
BM_OtherServices_df = pd.DataFrame({'OtherServices':BM})

# BM_Construction_df
BM = Construction['SelectionMethodDescription'].value_counts(normalize=True)
BM_Construction_df = pd.DataFrame({'Construction':BM})

# BM_GoodsServices_df
BM = GoodsServices['SelectionMethodDescription'].value_counts(normalize=True)
BM_GoodsServices_df = pd.DataFrame({'GoodsServices':BM})

# BM_ConstructionServices_df
BM = ConstructionServices['SelectionMethodDescription'].value_counts(normalize=True)
BM_ConstructionServices_df = pd.DataFrame({'ConstructionServices':BM})

# Combined Results
result = pd.concat([BM_goods_df, BM_HumanClientServices_df, BM_OtherServices_df, BM_Construction_df,BM_GoodsServices_df,BM_ConstructionServices_df], axis=1)
result.to_csv('Data/SelectionMethods.csv',index = True)

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


