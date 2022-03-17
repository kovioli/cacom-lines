# API data extraction
import requests
from wwo_hist import retrieve_hist_data
from datetime import datetime, timedelta, date
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import dateutil.relativedelta
import json
plt.style.use('fivethirtyeight')


# API electricity data
def get_elecprice(start_date,end_date,idx,freq_elec,token):
    ploads = {'things': 2, 'total': 25}
    r = requests.get('https://httpbin.org/get', params=ploads)
    url = 'https://api.montelnews.com/spot/getprices'
    authorisation='Bearer '+token
    headers = {
        'Authorization': authorisation}
    params = {'SpotKey': 14, 'Fields': [freq_elec], 'FromDate': start_date, 'ToDate': end_date, 'Currency': 'eur',
              'SortType': 'ascending'}
    r = requests.get(url, headers=headers, params=params)
    r_dictionary = r.json()
    df = pd.DataFrame(r_dictionary['Elements'])

    if freq_elec=='Base':
        df=df.set_index('Date')
        df=df.rename(columns={"Base": "elec_price"})
        df=df.drop(['TimeSpans'], axis=1)
        elec_price=df

    if freq_elec == 'Hours':
        # Create DataFrame
        # elec_price: data frame containing the price of elactricity each hour
        data = {'start_datetime': [], 'elec_price': []}
        elec_price = pd.DataFrame(data)
        # Add missing hour
        for index, row in df.iterrows():
            current_day = row['Date']
            current_timespan = row['TimeSpans']
            for i in current_timespan:
                current_time = i['TimeSpan'][0:5]
                Time1 = current_day[0:10] + ' ' + current_time + ':00'
                try:
                    start_datetime = datetime.strptime(Time1, '%Y-%m-%d %H:%M:%S')
                except:
                    hours_added = dt.timedelta(hours=1)
                    start_datetime = start_datetime + hours_added
                current_price = i['Value']
                elec_price.loc[len(elec_price.index)] = [start_datetime, current_price]
        elec_price['start_datetime'] = pd.to_datetime(elec_price['start_datetime'], infer_datetime_format='%Y-%m-%d %H:%M:%S')

        # add missing dates
        elec_price = elec_price.set_index('start_datetime')
        elec_price=elec_price.loc[~elec_price.index.duplicated(), :]
        elec_price = elec_price.reindex(idx, fill_value=np.nan)
        elec_price['elec_price'] = elec_price['elec_price'].interpolate(method='polynomial', order=2,limit_direction='both').ffill().bfill()
        elec_price = elec_price.drop(elec_price.index[len(elec_price) - 1])
    return elec_price

#extract oil prices
#you have a problem of interpolation
def get_oil_price(start_date,end_date,idx):
    brent_oil_prices = pd.read_excel('https://www.eia.gov/dnav/pet/hist_xls/RBRTED.xls', sheet_name='Data 1',skiprows=20, names=['Date', 'Brent_Prices'])
    brent_oil_prices = brent_oil_prices[(brent_oil_prices['Date'] >= start_date)]
    brent_oil_prices = brent_oil_prices[(brent_oil_prices['Date'] <= end_date)]
    #add missing dates
    brent_oil_prices = brent_oil_prices.set_index('Date')
    brent_oil_prices = brent_oil_prices.reindex(idx, fill_value=np.nan)
    brent_oil_prices['Brent_Prices']=brent_oil_prices['Brent_Prices'].interpolate(method='polynomial', order=2, limit_direction= 'both').ffill().bfill()
    brent_oil_prices = brent_oil_prices.drop(brent_oil_prices.index[len(brent_oil_prices) - 1])
    return brent_oil_prices

def get_gas_price(start_date,end_date,idx):
    gas_prices = pd.read_excel('https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls', sheet_name='Data 1', skiprows=2,names=['Date', 'Henry_Hub_Price'])
    gas_prices = gas_prices[(gas_prices['Date'] >= start_date)]
    gas_prices = gas_prices[(gas_prices['Date'] <= end_date)]
    #add missing dates
    gas_prices = gas_prices.set_index('Date')
    gas_prices = gas_prices.reindex(idx, fill_value=np.nan)
    gas_prices['Henry_Hub_Price']=gas_prices['Henry_Hub_Price'].interpolate(method='polynomial', order=2, limit_direction= 'both').ffill().bfill()
    gas_prices=gas_prices.drop(gas_prices.index[len(gas_prices)-1])
    return gas_prices

def get_wti_oil_price(start_date,end_date,idx):
    wti_prices =  pd.read_excel('https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls', sheet_name='Data 1', skiprows=2801, names=['Date', 'WTI_Prices'])
    wti_prices = wti_prices[(wti_prices['Date'] >= start_date)]
    wti_prices = wti_prices[(wti_prices['Date'] <= end_date)]
    #add missing dates
    wti_prices = wti_prices.set_index('Date')
    wti_prices = wti_prices.reindex(idx, fill_value=np.nan)
    wti_prices['WTI_Prices']=wti_prices['WTI_Prices'].interpolate(method='polynomial', order=2, limit_direction= 'both').ffill().bfill()
    wti_prices=wti_prices.drop(wti_prices.index[len(wti_prices)-1])
    return wti_prices

#extract weather infos
def get_weather(start_date,end_date,freq_weather):
    api_key = '468e02981d20483d88a181015212508'  # YOUR API KEY
    location_list = ['leipzig']
    frequency = freq_weather
    hist_weather_data = retrieve_hist_data(api_key,
                                           location_list,
                                           start_date,
                                           end_date,
                                           frequency,
                                           location_label=False,
                                           export_csv=True,
                                           store_df=True)
    leipzig_weather = pd.read_csv('leipzig.csv')
    leipzig_weather = leipzig_weather.drop(
        ['totalSnow_cm', 'moon_illumination', 'moonrise', 'moonset', 'sunrise', 'sunset', 'DewPointC', 'FeelsLikeC',
          'WindChillC', 'visibility', 'winddirDegree', 'location','pressure','precipMM','cloudcover','sunHour'], axis=1)
    leipzig_weather['date_time'] = pd.to_datetime(leipzig_weather['date_time'],     infer_datetime_format='%Y-%m-%d %H:%M:%S')
    leipzig_weather = leipzig_weather.set_index('date_time')
    return leipzig_weather

# Define a function to plot different types of time-series
def plot_series(df=None, column=None, series=pd.Series([]),label=None, ylabel=None, title=None, start=0, end=None):
    sns.set()
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.set_xlabel('Time', fontsize=16)
    if column:
        ax.plot(df[column][start:end], label=label)
        ax.set_ylabel(ylabel, fontsize=16)
    if series.any():
        ax.plot(series, label=label)
        ax.set_ylabel(ylabel, fontsize=16)
    if label:
        ax.legend(fontsize=16)
    if title:
        ax.set_title(title, fontsize=24)
    ax.grid(True)
    return ax

#function to check stability
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")


if __name__ == "__main__":



    url = 'https://coop.eikon.tum.de/mbt/mbt.json'
    r = requests.get(url)
    r_dictionary = r.json()
    token=r_dictionary['access_token']

    end_date = str(date.today())
    start_date= str(date.today() - timedelta(days=930))

    freq_elec='Hours'
    freq_weather = 1
    idx = pd.date_range(start_date, str(datetime.strptime(end_date, '%Y-%m-%d') + dt.timedelta(days=1)), freq='1H')
    freq_elec='Base'
    freq_weather=24
    idx = pd.date_range(start_date, str(datetime.strptime(end_date, '%Y-%m-%d') + dt.timedelta(days=1)), freq='D')
    elec_price= get_elecprice(start_date,end_date,idx,freq_elec,token)
    leipzig_weather = get_weather(start_date, end_date,freq_weather)
    oil_price=get_oil_price(start_date,end_date,idx)
    gas_price = get_gas_price(start_date, end_date,idx)
    wti_prices=get_wti_oil_price(start_date,end_date,idx)


    #check corrolation
    total_df=pd.concat([leipzig_weather, oil_price,gas_price,wti_prices,elec_price], axis=1)
    corr_matrix = total_df.corr()
    with open('corr.npy', 'wb') as f:
        np.save(f, corr_matrix)



