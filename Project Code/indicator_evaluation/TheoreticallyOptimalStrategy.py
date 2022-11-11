import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
import marketsimcode as marketsim
from util import get_data

def author():
    return 'rmikhael3'

def testTotal(df, sym):
    df['NextDayOffset'] = df.shift(-1)
    df["sum"] = df.sum(axis=1)
    Total = df["JPM"].sum()
    return df, Total

def createTrades(prices, sym):
    prices['NextDay'] = prices.shift(-1)
    trades = pd.DataFrame(index=prices.index)
    trades['Order'] = np.where(prices[sym] < prices['NextDay'], 'BUY', 'SELL')

    #cover short
    trades_cover = trades.copy()
    trades_cover['Order'] = trades_cover.shift(1)  #move it forward so when combined it covers it after reversing the operation
    trades_cover = trades_cover[1:]  #remove first day
    trades_cover['Temp'] = np.where(trades_cover['Order'] == 'BUY', 'SELL', 'BUY') #reverse it
    trades_cover.drop(['Order'], axis=1, inplace=True) #drop order to change name
    trades_cover.columns = ['Order']
    trade_df = pd.concat([trades, trades_cover]) #concat to fetch all operations
    trade_df.sort_index(inplace=True, ascending=True) #sort
    trade_df[sym] = np.where(trade_df['Order'] == 'BUY', 1000, -1000) #add order #
    trade_df = trade_df.loc[:, [sym]]
    return trade_df




def getPrices(symbol, start_date, end_date, Col=None):
    if Col == None:
        prices = get_data([symbol], pd.date_range(start_date, end_date), False)
    else:
        prices = get_data([symbol], pd.date_range(start_date, end_date), False, Col)

    prices = prices.fillna(method='ffill')
    prices = prices.fillna(method='bfill')
    prices_normed = prices / prices.iloc[0]
    return prices_normed

def testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):

    prices = getPrices(symbol, sd, ed)
    df_trade = createTrades(prices, symbol)
    #test1, test2 = testTotal(trade_df, symbol)
    return df_trade

def benchmarkTOS(symbol, sd, ed, shares):
    prices = getPrices(symbol, sd, ed)
    trades_df = prices.copy()
    trades_df[symbol] = 0
    start = trades_df.index.min()
    trades_df.loc[start, symbol] = shares
    return trades_df

def computeHoldingsFrame(pd, prices):
    holding_df_value = pd * prices
    return holding_df_value.sum(axis=1)

def cumRet(prices):
    return ((prices.iloc[-1] / prices.iloc[0]) - 1)

def dailyReturn(prices):
    dr = prices.pct_change()
    return dr[1:]

def Statistics(portvals = None):
    daily_returns = dailyReturn(portvals.copy())

    cum_ret = cumRet(portvals)
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    sharpe_ratio = np.sqrt(252.0) * (avg_daily_ret / std_daily_ret)

    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio

def tosVsBenchmark(df_trades_optimal, df_trades_benchmark, symbol = "JPM", sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31)):

    #Theoretically Optimal Strategy
    portvals_optimal = marketsim.compute_portvals(df_trades_optimal, start_val=100000, commission=0, impact=0)

    # Normalizing PortFolio Values
    portvals_normalized_optimal = portvals_optimal / portvals_optimal.iloc[0]
    cum_ret_optl, avg_daily_ret_optl, std_daily_ret_optl, sharpe_ratio_optl = Statistics(
        portvals_normalized_optimal)

    print(f"Date Range: {sd} to {ed} for {symbol}")
    print()
    print("Optimal Strategy")
    print(f"Sharpe Ratio of Fund: {sharpe_ratio_optl}")
    print(f"Cumulative Return of Fund: {cum_ret_optl}")
    print(f"Standard Deviation of Fund: {std_daily_ret_optl}")
    print(f"Average Daily Return of Fund: {avg_daily_ret_optl}")


    #Benchmark Strategy
    portvals_benchmark = marketsim.compute_portvals(df_trades_benchmark, start_val=100000, commission=0, impact=0)

    # Normalizing PortFolio Values
    portvals_normalized_benchmark = portvals_benchmark / portvals_benchmark.iloc[0]

    cum_ret_bench, avg_daily_ret_bench, std_daily_ret_bench, sharpe_ratio_bench = Statistics(portvals_normalized_benchmark)
    print()
    print("Benchmark")
    print(f"Sharpe Ratio of Fund: {sharpe_ratio_bench}")
    print(f"Cumulative Return of Fund: {cum_ret_bench}")
    print(f"Standard Deviation of Fund: {std_daily_ret_bench}")
    print(f"Average Daily Return of Fund: {avg_daily_ret_bench}")

    #TOS vs Benchamrk Strategy Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set(xlabel='Date', ylabel="Normalized Portfolio Value",
           title="Theoretically Optimal Strategy vs Benchmark Strategy")
    ax.plot(portvals_normalized_optimal, "red", label='Optimal Strategy')
    ax.plot(portvals_normalized_benchmark, "purple", label="Benchmark Strategy")
    ax.legend()
    fig.savefig('images/benchmarkVsTOS.png')
    plt.close()