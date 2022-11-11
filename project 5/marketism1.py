""""""
"""MC2-P1: Market simulator.  		  	   		  	  			  		 			     			  	 

Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			     			  	 
All Rights Reserved  		  	   		  	  			  		 			     			  	 

Template code for CS 4646/7646  		  	   		  	  			  		 			     			  	 

Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			     			  	 
or edited.  		  	   		  	  			  		 			     			  	 

We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			     			  	 
GT honor code violation.  		  	   		  	  			  		 			     			  	 

-----do not edit anything above this line---  		  	   		  	  			  		 			     			  	 

Student Name: Rimon Mikhael (replace with your name)  		  	   		  	  			  		 			     			  	 
GT User ID: rmikhael3 (replace with your User ID)  		  	   		  	  			  		 			     			  	 
GT ID: 903737444 (replace with your GT ID)  		  	   		  	  			  		 			     			  	 
"""

import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data


def getSymbols(orders_file):
    df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    df.sort_index(inplace=True)
    return list(df["Symbol"].unique()), df


def computeTradesFrame(sym, df, impact, commission, start_val):
    start_date = df.index.min()
    end_date = df.index.max()
    prices = get_data(sym, pd.date_range(start_date, end_date))
    prices.ffill(inplace=True)  # Forward Filling first
    prices.bfill(inplace=True)

    if 'SPY' not in sym:
        prices.drop('SPY', axis=1, inplace=True)

    prices['Cash'] = 1

    trades_df = pd.DataFrame(np.zeros(prices.shape))
    trades_df.index = prices.index
    trades_df.columns = prices.columns
    trades_df.iloc[0, -1] = start_val

    for index, row in df.iterrows():
        stock = row['Symbol']
        type = row['Order']
        shares = row['Shares']
        sym_price = prices.loc[index, stock]  # getting stock price from prices df

        if type == 'SELL':
            multiplier = -1
            trades_df.loc[index, stock] = trades_df.loc[index, stock] - shares
            sym_price = sym_price - (sym_price * impact)
        else:
            multiplier = 1
            trades_df.loc[index, stock] = trades_df.loc[index, stock] + shares
            sym_price = sym_price + (sym_price * impact)

        trades_df.loc[index, 'Cash'] = trades_df.loc[index, 'Cash'] - commission - (sym_price * shares * multiplier)

    return trades_df, prices


def computeHoldingsFrame(pd, prices):
    holding_df_value = pd * prices
    return holding_df_value.sum(axis=1)


def compute_portvals(
        orders_file="./orders/orders.csv",
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

    sym, df = getSymbols(orders_file)

    trades_df, prices = computeTradesFrame(sym, df, impact, commission, start_val)

    portvals = computeHoldingsFrame(trades_df.cumsum(), prices)

    return portvals


def author():
    return 'rmikhael3'


def cumRet(prices):
    return ((prices.iloc[-1] / prices.iloc[0]) - 1)


def dailyReturn(prices):
    dr = prices.pct_change()
    return dr[1:]


def Statistics(portvals=None, mode=True, start_date=0, end_date=0):
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


def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-02.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[
            0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    orders_temp = pd.read_csv(of, index_col='Date', parse_dates=True, na_values=['nan'])
    start_date = orders_temp.index.min()
    end_date = orders_temp.index.max()

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = Statistics(portvals=portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = Statistics(start_date=start_date,
                                                                                     end_date=end_date, mode=False)

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
