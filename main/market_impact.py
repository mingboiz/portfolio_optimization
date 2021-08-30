import numpy as np
import pandas as pd
import cvxpy as cp
import yfinance as yf
import pypfopt
import pathlib
import sklearn
import time
import tqdm
from pathlib import Path
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
np.random.seed(1)

# filter out daily DAV of t-1
def one_year_DAV(df, tickers, t):
    '''
    Returns a pd.Series of the one-year Daily Average Volume of given ticker(s) from time t
    
    Parameters:
        df::DataFrame: SEP.csv, with a 'date' DateTime64 column or similar
        tickers::[Strings]: List of strings of names of ticker(s) to get the previous year DAV
        t::pd.Timestamp or similar: Date to filter on
        timedelta::str: 'd, day, days' or 'y, year, years' get DAV of previous day/year
    
    Returns:
        dav::float: average one-year daily volume
    '''
    timedelta = relativedelta(years=1)
    # get range of prev_period to t
    prev_period = t - timedelta
    
    if t not in df['date'].values:
        return 'No data for {t}'.format(t=t)
    
    if not df['ticker'].isin(tickers).any():
        return 'No ticker for any of {tickers}'.format(tickers="tickers")

    if isinstance(df.index, pd.DatetimeIndex):
        mask = (df['ticker'].isin(tickers)) & \
               (df.loc[df.index >= prev_period]) & \
               (df.loc[df.index < t])
    else:
        mask = (df['ticker'].isin(tickers)) & \
               (df['date'] >= prev_period) & \
               (df['date'] < t)
    
    temp_df = df[mask]
    dav = temp_df.groupby('ticker').mean()['volume']
    return dav

def theta_2(t, coef=0.01):
    '''
    Returns theta_2 * the value of the time_trend, 
    where June 1926 is 1 and July 1926 is 2, ...
    
        Parameters:
            t::pd.Timestamp or similar
            coef::float: Coefficient of the time_trend, defaults to 0.01
        
        Returns:
            diff_in_months::float: coef * time_trend
    '''
    start_time = pd.Timestamp('1926-06-01')
    curr_time = pd.Timestamp(t)
    if curr_time.month >= start_time.month:
        diff_in_years = curr_time.year - start_time.year
    else:
        diff_in_years = curr_time.year - 1 - start_time.year
    #return ( coef * diff_in_years ) # uncomment this line to stop debug
    return diff_in_years

def theta_3(df, tickers, t, coef=-0.14):
    '''
    Returns theta_3 * log(1 + market cap of equity (USD Billions))

        Parameters:
            df:DataFrame: DAILY.csv, with a 'date' DateTime64 column or similar
            tickers::[Strings]: List of strings of names of ticker(s) to get the previous year DAV
            t::pd.Timestamp or similar: Date to filter on

        Returns:
            ans::float: log ( 1 + market cap of equity (in billions of USD))

    '''
    timedelta = relativedelta(days=1)
    # get range of prev_period to t
    prev_period = t - timedelta

    if prev_period not in df['date'].values:
        print(prev_period, 'not available')
        # This is a catch for long weekends/holidays where there is no available data
        for i in range(4):
            prev_period = prev_period - relativedelta(days = 1)
            if prev_period in df['date'].values:
                print("found data for prev_period:", prev_period)
                break
    print('\nnew prev_period:', prev_period)

    if ~df['ticker'].isin(tickers).any():
        return 'No ticker for any of {tickers}'.format(tickers="tickers")

    mask = (df['ticker'].isin(tickers)) & (df['date'] == prev_period)

    temp_df = df[mask].drop_duplicates()
    market_value_of_equity = pd.Series(index = temp_df['ticker'], 
                           data = (temp_df[['marketcap']].values/1000).flatten(), 
                           name = 'marketcap')
    market_value_of_equity.fillna(1e-5, inplace=True)
    #print("market_cap\n", market_value_of_equity)
    tickers_series = pd.Series(data=tickers, name='tickers')
    #print("tickers\n", tickers_series)
    temp = pd.merge(market_value_of_equity, tickers_series, left_index=True, right_on='tickers', how='inner').drop_duplicates()
    market_value_of_equity = pd.Series(data = temp['marketcap'].values, index=temp['tickers'], name='market_value_of_equity', dtype='float64')

    if len(market_value_of_equity) < len(tickers_series):
        print("Theta_3: {tickers} Market Cap not returning properly".format(tickers = set(tickers) ^ set(market_value_of_equity.index)))

        missing_tickers = list(set(tickers) ^ set(market_value_of_equity.index))
        marketcap_missing_tickers = []
        for ticker in missing_tickers:
            marketcap_missing_tickers.append(yf.Ticker(ticker).info['marketCap'] / 1000000000)

        missing_data = pd.Series(data=marketcap_missing_tickers, index=missing_tickers, name='marketcap_missing_tickers')
        market_value_of_equity = pd.concat([market_value_of_equity, missing_data], axis=0)
    #market_value_of_equity.drop_duplicates(inplace=True)
    try:
        ans = np.log(1 + market_value_of_equity)
        return ( ans * coef )
    except TypeError:
        print('\n')
        print("Check market_cap function returns a float")

def theta_4_and_5(df, tickers, t, curr_df, prev_df, coef=(-0.53, 11.21)):
    '''
    Returns theta_4, theta_5: (Fraction of Daily Volume of the Trade, Sqrt(Fraction of Daily Volume of Trade)
    
    ====================== Need to figure out how to pass prev_df =============================
    
    Parameters:
        df::DataFrame: SEP.csv, with a Datetime64 'date' column
        tickers::[Strings]: List of strings of names of ticker(s) to get the previous year DAV
        t::pd.Timestamp or similar:
        curr_df::DataFrame: Current DataFrame with a 'long'(bool), 'price', 'shares' column
        prev_df::DataFrame: Previous DataFrame with a 'long'(bool), 'price', 'shares' column 
        coef::float: coefficient in Table VIII

    Returns:
        Fraction of daily volume: coef * Trade's dollar size / stock's 1-year DAV (in %)
    '''
    davs = one_year_DAV(df, tickers, t)

    temp_df = pd.merge(davs, prev_df, left_index=True, right_on='ticker', how='inner')
    temp_df = pd.merge(temp_df, curr_df.set_index('ticker')[['long', 'shares', 'price']], left_on='ticker', right_index=True,
                      how='inner', suffixes=("_prev", "_curr"))
    #print(temp_df.shape)

    size_of_trades = np.zeros(len(temp_df))
    
    for i in range(len(temp_df)):
            new_shares = temp_df.iloc[i]['shares_curr'] - temp_df.iloc[i]['shares_prev']
            size_of_trades[i] = new_shares * temp_df.iloc[i]['price_curr']

    size_of_trade = pd.Series(data = size_of_trades, index = temp_df['ticker'], name = 'size_of_trade')

    temp_df = pd.merge(temp_df, size_of_trade, left_on='ticker', right_index=True, how='inner')
    temp_df['frac_of_daily_volume'] = np.where(temp_df['size_of_trade'] != 0, temp_df['size_of_trade'] / temp_df['volume'], 0)
    temp_df['sqrt_of_frac_of_daily_volume'] = np.sqrt(np.abs(temp_df['frac_of_daily_volume']))
    
    temp_df.sort_index(inplace=True)
    temp_df.set_index('ticker', inplace=True)
    theta_4 = coef[0] * temp_df['frac_of_daily_volume'] 
    theta_5 = coef[1] * temp_df['sqrt_of_frac_of_daily_volume']
    
    # return theta_4, theta_5 # uncomment this line to stop debug
    return temp_df

def theta_6(df, benchmark, tickers, t, coef=0.31):
    '''
    Returns a pd.Series of coef * idiosyncratic volatility of stocks
    
    It is calculated by regressing one-year daily stock returns against the MSCI-US benchmark, 
    then the idiosyncratic volatility is the standard deviation of the residuals 
    
    Parameters:
        df::DataFrame: SEP.csv with a 'Date' DateTime64 column or similar
        benchmark::DataFrame: mscius.xlsx with 'Exchange Date' DateTime64 column and 'Close' column
        tickers::[Strings]: List of strings of names of ticker(s) to get the previous year DAV
        t::pd.Timestamp or similar: Date to filter on
        coef::float: theta_6 value
    
    Returns:
        ans::float: Idiosyncratic Volatility of regression of one-year daily stock returns on MSCI-US
    
    benchmark = alpha + beta * instrument returns + residuals
    
    Calculates the idiosyncratic volatility of a stock, which measures
    how much an investment's price fluctuates around relative
    to its relationship to a benchmark. 
    
    It is directly tied to beta, which is a measure of whether
    a given investment's fluctuations are larger or smaller
    than that of another index. 
    '''
    from sklearn.linear_model import LinearRegression
       
    prev_period = t - relativedelta(years=1)
    
    #mask = (df['ticker'].isin(tickers)) & (df['date'] >= prev_period) & (df['date'] < t)
    mask = (df['ticker'].isin(tickers)) & (df['date'] == prev_period)
    if df[mask].shape[0] < 1:
        print('No data for {t} in Data'.format(t=prev_period))
    for i in range(4):
        prev_period = prev_period - relativedelta(days = 1)
        mask = (df['date'] == prev_period) & (df['ticker'].isin(tickers))
        if df[mask].shape[0] == len(tickers):
            print("found new prev_period with all data:", prev_period)
            break
        
    print('\nNew prev_period', prev_period, 't', t)
    mask = (df['ticker'].isin(tickers)) & (df['date'] >= prev_period) & (df['date'] < t)
    prices_df = df[mask].sort_values('date', axis=0, ascending=True, ignore_index=True)
    prices_df['returns'] = prices_df.groupby(['ticker'])['close'].pct_change()
    returns_df = prices_df[['date', 'ticker', 'returns']].set_index('date').fillna(0)
    #print('returns_df', returns_df)
    
    mask = (benchmark.iloc[:,0] >= prev_period) & (benchmark.iloc[:,0] < t)
    if benchmark[mask].shape[0] < 1:
        return 'No data for {t} in {df}'.format(t=t, df=df)
    benchmark_df = benchmark[mask].sort_values('Exchange Date', axis=0, ascending=True, ignore_index=True).set_index('Exchange Date')
    benchmark_df = benchmark_df['Close'].pct_change().fillna(0)
    #print("benchmark_df", benchmark_df)
    volatility_df = pd.merge(returns_df, benchmark_df, how='inner', left_index=True, right_index=True, validate='m:1')
    #print("volatility_df", volatility_df)
    idiosyncratic_volatility_list = []
    for ticker in tickers:
        mask = volatility_df['ticker'] == ticker
        X = volatility_df[mask]['returns'].values.reshape(-1,1)
        y = volatility_df[mask]['Close'].values
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)
        residuals = preds - y
        idiosyncratic_volatility_list.append( np.std(residuals, ddof=1) )

    idiosyncratic_volatility = pd.Series(data=idiosyncratic_volatility_list, index=tickers, name="idiosyncratic_volatility")
    #idiosyncratic_volatility.drop_duplicates(inplace=True)
    
    #return ( idiosyncratic_volatility * coef ) # uncomment this line to stop debug
    return idiosyncratic_volatility

def theta_7(crsp, t, coef=0.12):
    '''
    Returns theta_7 * monthly variance of CRSP-value weighted index    

    Parameters:
        df:DataFrame: CRSP-Value Index with a DateTime64 first column or similar
        coef::float: theta_7 value
        t::pd.Timestamp or similar: Date to filter on

    Returns:
        ans::float: Sample monthly variance of CRSP-Value Weighted Index
    '''

    past_month = t - relativedelta(months=1)
    mask = (crsp.iloc[:,0] >= past_month) & (crsp.iloc[:,0] < t)
    if crsp[mask].shape[0] < 1:
        return 'No data for {t}'.format(t=t)
    vix_values = crsp[mask].iloc[:,1].pct_change().fillna(0)
    # vix = np.var(crsp_values, ddof=1)
    vix = np.median(vix_values)
    #return ( vix * coef ) # uncomment this to stop debug
    return vix_values

def market_impact(daily, sep, benchmark, vix, list_of_tickers, currdf, prevdf, t):
    """
        Returns the market impact of a trade on the market

            Parameters:
                daily::pd.DataFrame: DAILY.csv
                sep::pd.DataFrame: SEP.csv
                benchmark::pd.DataFrame: mscius.xlsx ('Exchange Date', 'Close')
                crsp::pd.DataFrame: crsp.xlsx ('Exchange Date', Close')
                ticker::[Strings]: List of strings of names of ticker(s)
                t::pd.Timestamp: Date of rebalance
                currdf::DataFrame: Current DataFrame with a 'long'(bool), 'price', 'shares' column
                prevdf::DataFrame: Previous DataFrame with a 'long'(bool), 'price', 'shares' column 


            Returns:
                market_impact::float: Estimated Market Impact of the trade   

        Market Impact Model uses estimated coefficients from Table VII of Frazzini et al - Trading Costs (2018)
        Column (5) for a Global sample, Column (10) for US sample, Column (15) for International

        MI = a + b*x + c*sign(x)*sqrt( abs(x) )
        x = m/dtv*100%, where m = sign(dollar volume) and dtv =  stock's average one year dollar volume
        dollar volume: total value of shares traded = volume * price
        
    """
    #print(list_of_tickers)
    # Theta_2: Time trend
    theta_2 = -0.01
    
    start_time = pd.Timestamp('1926-06-01')
    curr_time = pd.Timestamp(t)
    if curr_time.month >= start_time.month:
        diff_in_years = curr_time.year - start_time.year
    else:
        diff_in_years = curr_time.year - 1 - start_time.year
    
    theta2 = theta_2 * diff_in_years
    
    # Theta_3: Log (1 + market cap)
    theta_3 = -0.14
    timedelta = relativedelta(days=1)
    # get range of prev_period to t
    prev_period = t - timedelta
    
    if prev_period not in daily['date'].values:
        #print(prev_period, 'not available')
        # This is a catch for long weekends/holidays where there is no available data
        for i in range(4):
            prev_period = prev_period - relativedelta(days = 1)
            if prev_period in daily['date'].values:
                print("Theta_3: found data for prev_period:", prev_period)
                break
    #print('\nTheta_3 new prev_period:', prev_period)
    
    if ~daily['ticker'].isin(list_of_tickers).any():
        return 'Theta_3: No ticker for any of {tickers}'.format(tickers="tickers")
       
    mask = (daily['ticker'].isin(list_of_tickers)) & (daily['date'] == prev_period)
    
    temp_df = daily[mask].drop_duplicates()
    market_value_of_equity = pd.Series(index = temp_df['ticker'], 
                           data = (temp_df[['marketcap']].values/1000).flatten(), 
                           name = 'marketcap', dtype='float64')
    market_value_of_equity.fillna(1e-5, inplace=True)
    #print("market_cap\n", market_value_of_equity)
    tickers_series = pd.Series(data=list_of_tickers, name='tickers')
    #print("tickers\n", tickers_series)    
    temp = pd.merge(market_value_of_equity, tickers_series, left_index=True, right_on='tickers', how='inner').drop_duplicates()
    market_value_of_equity = pd.Series(data = temp['marketcap'].values, index=temp['tickers'], name='market_value_of_equity', dtype='float64')
    
    if len(market_value_of_equity) < len(list_of_tickers) and len( set(list_of_tickers) ^ set(market_value_of_equity.index) ) > 0:
        print("Theta_3: {tickers} data not available in DAILY.csv".format(tickers = set(list_of_tickers) ^ set(market_value_of_equity.index)))
 
        missing_tickers = list(set(list_of_tickers) ^ set(market_value_of_equity.index))
        marketcap_missing_tickers = []
        # Download the latest market cap from Yahoo Finance
        for ticker in missing_tickers:
            marketcap_missing_tickers.append(yf.Ticker(ticker).info['marketCap'] / 1000000000)

        missing_data = pd.Series(data=marketcap_missing_tickers, index=missing_tickers, name='marketcap_missing_tickers', dtype='float64')
        market_value_of_equity = pd.concat([market_value_of_equity, missing_data], axis=0) 

    try:
        ans = np.log(1 + market_value_of_equity)
    except TypeError:
        print('\n')
        print("Theta_3: Check market_cap function returns a float")
    theta3 = ans * theta_3
    
    # Theta_4 and Theta_5: Fraction of Daily Volume and Sqrt(Fraction of Daily Volume)
    theta_4, theta_5 = -0.53, 11.21
    
    timedelta = relativedelta(years=1)
    # get range of prev_period to t
    try:
        prev_period = t - timedelta
    except:
        print("Theta_4_5: t is not a Datetime", t)
    #print(list_of_tickers)
    mask = (sep['ticker'].isin(list_of_tickers)) & (sep['date'] == prev_period)
    if sep[mask].shape[0] < 1:
        #print('Theta_4_5_6:No data for {t} in Data'.format(t=prev_period))
        for i in range(4):
            prev_period = prev_period - relativedelta(days = 1)
            mask = (sep['date'] == prev_period) & (sep['ticker'].isin(list_of_tickers))
            if sep[mask].shape[0] == len(list_of_tickers):
                print("Theta_4_5_6: found new prev_period with all data:", prev_period)
                break
        
    print('\nNew prev_period', prev_period, 't', t)

    if t not in sep['date'].values:
        return 'Theta_4, Theta_5:No data for {t}'.format(t=t)
    
    if not sep['ticker'].isin(list_of_tickers).any():
        return 'Theta_4, Theta_5: No ticker for any of {tickers}'.format(tickers="tickers")
    
    mask = (sep['ticker'].isin(list_of_tickers)) & \
           (sep['date'] >= prev_period) & \
           (sep['date'] < t)
    
    temp_df = sep[mask]
    
    # Theta_6: Idiosyncratic Volatility
    # uses the same dataframe as Theta_4 and Theta_5 so I put it here for efficiency
    theta_6 = 0.31
    if temp_df.shape[0] < 1:
        return 'Theta_6: No data for {t} in data'.format(t=t)
    prices_df = temp_df.sort_values('date', axis=0, ascending=True, ignore_index=True)
    prices_df['returns'] = prices_df.groupby(['ticker'])['close'].pct_change()
    returns_df = prices_df[['date', 'ticker', 'returns']].set_index('date').fillna(0)

    mask = (benchmark.iloc[:,0] >= prev_period) & (benchmark.iloc[:,0] < t)
    if benchmark[mask].shape[0] < 1:
        return 'Theta_6: No data for {t} in data'.format(t=t)
    benchmark_df = benchmark[mask].sort_values('Exchange Date', axis=0, ascending=True, ignore_index=True).set_index('Exchange Date')
    benchmark_df = benchmark_df['Close'].pct_change().fillna(0) # Close is now returns
    volatility_df = pd.merge(returns_df, benchmark_df, how='inner', left_index=True, right_index=True, validate='m:1')
  
    idiosyncratic_volatility_list = []
    for ticker in list_of_tickers:
        mask = volatility_df['ticker'] == ticker
        X = volatility_df[mask]['returns'].values.reshape(-1,1)
        y = volatility_df[mask]['Close'].values
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)
        residuals = preds - y
        idiosyncratic_volatility_list.append( np.std(residuals, ddof=1) )

    idiosyncratic_volatility = pd.Series(data=idiosyncratic_volatility_list, index=list_of_tickers, name="idiosyncratic_volatility", dtype='float64')
    idiosyncratic_volatility.drop_duplicates(inplace=True)
    theta6 = theta_6 * idiosyncratic_volatility
    
    # Theta_4 and Theta_5: Fraction of Trade of 1-year Daily Average Volume
    
    davs = temp_df.groupby('ticker').mean()['volume']

    temp_df = pd.merge(davs, prevdf, left_index=True, right_on='ticker', how='inner')
    temp_df = pd.merge(temp_df, currdf.set_index('ticker')[['long', 'shares', 'price']], left_on='ticker', right_index=True,
                      how='inner', suffixes=("_prev", "_curr"))
    #temp_df.drop_duplicates(inplace=True) # not sure if this affects anything if error remove this comment

    size_of_trades = np.zeros(len(temp_df))
    
    for i in range(len(temp_df)):
        new_shares = temp_df.iloc[i]['shares_curr'] - temp_df.iloc[i]['shares_prev']
        size_of_trades[i] = new_shares * temp_df.iloc[i]['price_curr']

    size_of_trade = pd.Series(data = size_of_trades, index = temp_df['ticker'], name = 'size_of_trade', dtype='float64')

    temp_df = pd.merge(temp_df, size_of_trade, left_on='ticker', right_index=True, how='inner')
    temp_df['frac_of_daily_volume'] = np.where(temp_df['size_of_trade'] != 0,  temp_df['size_of_trade'] / temp_df['volume'], 0)
    temp_df['sqrt_of_frac_of_daily_volume'] = np.sqrt(np.abs(temp_df['frac_of_daily_volume']))
    
    temp_df.sort_index(inplace=True)
    temp_df.set_index('ticker', inplace=True)
    theta4 = theta_4 * temp_df['frac_of_daily_volume'] 
    theta5 = theta_5 * temp_df['sqrt_of_frac_of_daily_volume']
    
    # Theta_7
    theta_7 = 0.12
    
    prev_period = t - relativedelta(months=1)
    mask = (vix.iloc[:,0] >= prev_period) & (vix.iloc[:,0] < t)
    if vix[mask].shape[0] < 1:
        return 'No data for {t}'.format(t=t)
    vix_values = vix[mask].iloc[:,1].pct_change().fillna(0)
    
    theta7 = theta_7 * vix_values
    
    a = theta2 + np.median(theta3) + np.median(theta6) + np.median(theta7)
    #a = np.median(theta3) + np.median(theta6) + np.median(theta7)
    b = theta4
    c = theta5
    marketimpact = a + b + c
    #print('\ntheta_2', theta2, '\ntheta3\n', theta3, '\ntheta4\n', theta4, '\ntheta5\n', theta5, '\ntheta6\n', theta6, '\ntheta7\n', theta7, '\n')
    return marketimpact