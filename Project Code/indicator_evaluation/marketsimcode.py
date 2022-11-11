import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data

def computeTradesFrame(syms, df, impact, commission, start_val):
    start_date = df.index.min()
    end_date = df.index.max()
    prices = get_data(syms, pd.date_range(start_date, end_date), False)
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)


    prices['Cash'] = 1

    trades_df = pd.DataFrame(np.zeros(prices.shape))
    trades_df.index = prices.index
    trades_df.columns = prices.columns
    trades_df.iloc[0, -1] = start_val

    for sym in syms:
        for index, row in df.iterrows():
            stock = sym
            shares = row[sym]
            sym_price = prices.loc[index, stock]  # getting stock price from prices df

            if shares < 0:
                trades_df.loc[index, stock] = trades_df.loc[index, stock] + shares
                sym_price = sym_price - (sym_price * impact)

            else:
                trades_df.loc[index, stock] = trades_df.loc[index, stock] + shares
                sym_price = sym_price + (sym_price * impact)

            trades_df.loc[index, 'Cash'] = trades_df.loc[index, 'Cash'] - commission - (sym_price * shares)

    return trades_df, prices

def computeHoldingsFrame(pd, prices):
    holding_df_value = pd * prices
    return holding_df_value.sum(axis=1)

def cumRet(prices):
    return ((prices.iloc[-1] / prices.iloc[0]) - 1)

def dailyReturn(prices):
    dr = prices.pct_change()
    return dr[1:]

def Statistics(portvals = None, mode = True, start_date = 0, end_date = 0):
    if (mode):
        daily_returns = dailyReturn(portvals.copy())

        cum_ret = cumRet(portvals)
        avg_daily_ret = daily_returns.mean()
        std_daily_ret = daily_returns.std()
        sharpe_ratio = np.sqrt(252.0) * (avg_daily_ret / std_daily_ret)

        return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio

    else:
        prices = get_data(['$SPX'], pd.date_range(start_date, end_date))
        del prices['SPY']

        daily_returns = dailyReturn(prices.copy())

        cum_ret = cumRet(prices)
        avg_daily_ret = daily_returns.mean()
        std_daily_ret = daily_returns.std()
        sharpe_ratio = (np.sqrt(252.0) * (avg_daily_ret / std_daily_ret))

        return cum_ret.item(), avg_daily_ret.item(), std_daily_ret.item(), sharpe_ratio.item()

def getSymbols(orders_df):
    df = orders_df
    df.sort_index(inplace=True)
    return list(orders_df.columns), df

def author():
    return 'rmikhael3'

def compute_portvals(
        tradesDf,
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    syms, df = getSymbols(tradesDf)

    trades_df, prices = computeTradesFrame(syms, df, impact, commission, start_val)

    portvals = computeHoldingsFrame(trades_df.cumsum(), prices)

    return portvals

