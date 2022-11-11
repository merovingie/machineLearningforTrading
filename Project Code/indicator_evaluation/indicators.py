import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data
import warnings
warnings.filterwarnings("ignore")

def author():
    return 'rmikhael3'

def getSymbols(orders_df):
    df = orders_df
    df.sort_index(inplace=True)
    return list(orders_df.columns), df

def getPrices(symbol, start_date, end_date, Col=None):
    if Col == None:
        prices = get_data([symbol], pd.date_range(start_date, end_date), False)
    else:
        prices = get_data([symbol], pd.date_range(start_date, end_date), False, Col)

    prices = prices.fillna(method='ffill')
    prices = prices.fillna(method='bfill')
    prices_normed = prices / prices.iloc[0]
    return prices_normed

# Simple Moving Average
def simpleMovingAverage(prices, window):
    return prices.rolling(window=window).mean()

def mySMA(prices, window):
    sma_price = simpleMovingAverage(prices, window) / prices
    sma_20 = simpleMovingAverage(prices, 20)
    return simpleMovingAverage(prices, window), sma_price, sma_20

# Bollinger Bands
def bollingerBands(prices, window):
    sma = prices.rolling(window).mean()
    stdev = prices.rolling(window).std()
    upperband = sma + (2 * stdev)
    lowerband = sma - (2 * stdev)
    BB_value = (prices - lowerband) / (upperband - lowerband)
    return upperband, lowerband, BB_value



# MACD
def movingAverageConvergenceDivergence(prices):
    ema_fast = prices.ewm(span=12, adjust=False).mean()
    ema_slow = prices.ewm(span=26, adjust=False).mean()
    MACD = ema_slow - ema_fast
    signal = MACD.ewm(span=9, adjust=False).mean()
    return MACD, signal

# Momentum
def momentum(prices, window):
    return prices / prices.shift(window) - 1


# Stochastic
def stochastic(sym, window, sd, ed):
    prices_high = getPrices(sym, sd, ed, 'High')
    prices_high.rename(columns={prices_high.columns[0]:'Date', sym: 'High'}, inplace=True)
    prices_low = getPrices(sym, sd, ed, 'Low')
    prices_low.rename(columns={prices_low.columns[0]:'Date', sym: 'Low'}, inplace=True)
    prices_close = getPrices(sym, sd, ed)
    prices_close.rename(columns={prices_close.columns[0]:'Date', sym: 'Close'}, inplace=True)
    prices_high = prices_high.join(prices_low)
    prices_high = prices_high.join(prices_close)
    prices_all = prices_high.copy()
    prices_all['st_high'] = prices_all['High'].rolling(window).max()
    prices_all['st_low'] = prices_all['Low'].rolling(window).min()
    prices_all['%K'] = (prices_all['Close'] - prices_all['st_low']) * 100 / (prices_all['st_high'] - prices_all['st_low'])
    prices_all['%D'] = prices_all['%K'].rolling(3).mean()
    return prices_all['%K'], prices_all['%D']




def graphIndicators(symbol = "JPM", sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31)):

    window = 14
    prices = getPrices(symbol, sd, ed)

    # SMA
    sma, sma_by_price, sma_50_days = mySMA(prices, window)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set(xlabel='Time', ylabel="Price", title="Simple Moving Average")
    ax.plot(prices, "red", label="Normalized Price")
    ax.plot(sma, "purple", label="SMA")
    ax.legend(loc="best")
    fig.savefig('images/SMA.png')
    plt.close()

    # Bollinger Bands
    upperband, lowerband, BBP = bollingerBands(prices, window)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set(xlabel='Time', ylabel="Price", title="Bollinger Bands")
    ax.plot(prices, "red", label='Normalized Price')
    ax.plot(sma, "purple", label="Rolling Average")
    ax.plot(upperband, "blue", label="Upper Band")
    ax.plot(lowerband, "yellow", label="Lower Band")
    ax.legend()
    fig.savefig('images/BB.png')
    plt.close()

    # Bollinger Band %
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set(xlabel='Time', ylabel="Price", title="Bollinger Bands Percentage")
    ax.plot(prices, "red", label='Normalized Price')
    ax.plot(BBP, "blue", label='BBP%')
    ax.legend()
    fig.savefig('images/BBP.png')
    plt.close()

    # Momentum
    momn = momentum(prices, window)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set(xlabel='Time', ylabel="Price", title="Momentum")
    ax.plot(prices, "red", label='Normalized Price')
    ax.plot(momn, "blue", label="Momentum")
    ax.legend()
    fig.savefig('images/Momentum.png')
    plt.close()

    # Stochastic
    K, D = stochastic(symbol, window, sd, ed)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set(xlabel='Time', ylabel="change of price", title="Stochastic")
    plt.ylim(0,100)
    ax.plot(K, "blue", label='stochastic %K')
    ax.plot(D, "red", label="stochastic %D")
    ax.axhline(20, linestyle='--', color="black")
    ax.axhline(80, linestyle="--", color="black")
    ax.legend()
    fig.savefig('images/stochastic.png')
    plt.close()

    # MACD
    MACD, signal = movingAverageConvergenceDivergence(prices)
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set(xlabel='Time', ylabel="Price", title="MACD")
    ax.plot(MACD, "blue", label='MACD')
    ax.plot(signal, "red", label="MACD Signal")
    ax.legend()
    fig.savefig('images/MACD.png')
    plt.close()

    # INdicators Debugging testing
    # stochastic(symbol, window, sd, ed)
    # print('DebugSTOP')

