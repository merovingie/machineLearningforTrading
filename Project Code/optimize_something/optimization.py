""""""  		  	   		  	  			  		 			     			  	 
"""MC1-P2: Optimize a portfolio.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
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
import numpy as np
import matplotlib.pyplot as plt  		  	   		  	  			  		 			     			  	 
import pandas as pd  		  	   		  	  			  		 			     			  	 
from util import get_data, plot_data
from scipy import optimize as spo

def getMaxSharpe(prices, allocs):
    """
    using negative to get maximize from spo.optimize.minimize
    """

    neg_sharpe_ratio = -1 * getSharpe(prices, allocs)
    return neg_sharpe_ratio

def getSharpe(prices, allocs):
    """
    Find the best allocation for a portfolio
    based-on the Maximum Sharpe ratio of the stocks in it.
    Setting boundary to be (0, 1);
    Constraints to be sum(allocations) == 1
    """
    risk_free_return = 0
    sampling_period = 252
    prices_allcs = prices * allocs
    row_val_df = prices_allcs.sum(axis=1)
    daily_return = row_val_df.pct_change()
    #avg_daily_return = daily_return[1:].mean()
    sharpe = (np.sqrt(sampling_period) * (daily_return[1:] - risk_free_return).mean() / ((daily_return[1:] - risk_free_return).std()))
    return sharpe

# This is the function that will be tested by the autograder  		  	   		  	  			  		 			     			  	 
# The student must update this code to properly implement the functionality
def optimize_portfolio(  		  	   		  	  			  		 			     			  	 
    sd=dt.datetime(2008, 1, 1),  		  	   		  	  			  		 			     			  	 
    ed=dt.datetime(2009, 1, 1),  		  	   		  	  			  		 			     			  	 
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		  	  			  		 			     			  	 
    gen_plot=False,  		  	   		  	  			  		 			     			  	 
):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		  	  			  		 			     			  	 
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		  	  			  		 			     			  	 
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		  	  			  		 			     			  	 
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		  	  			  		 			     			  	 
    statistics.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			     			  	 
    :type sd: datetime  		  	   		  	  			  		 			     			  	 
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			     			  	 
    :type ed: datetime  		  	   		  	  			  		 			     			  	 
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		  	  			  		 			     			  	 
        symbol in the data directory)  		  	   		  	  			  		 			     			  	 
    :type syms: list  		  	   		  	  			  		 			     			  	 
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		  	  			  		 			     			  	 
        code with gen_plot = False.  		  	   		  	  			  		 			     			  	 
    :type gen_plot: bool  		  	   		  	  			  		 			     			  	 
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		  	  			  		 			     			  	 
        standard deviation of daily returns, and Sharpe ratio  		  	   		  	  			  		 			     			  	 
    :rtype: tuple  		  	   		  	  			  		 			     			  	 
    """
    # Read in adjusted closing prices for given symbols, date range  		  	   		  	  			  		 			     			  	 
    dates = pd.date_range(sd, ed)  		  	   		  	  			  		 			     			  	 
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		  	  			  		 			     			  	 
    prices = prices_all[syms]  # only portfolio symbols  		  	   		  	  			  		 			     			  	 
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

    """
    Portoflio
    """
    # dealing with missing values in the data file.
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

    #calculate basic states before calling maximize function
    norm_prices = prices / prices.ix[0, :]  # normalized prices
    size = len(prices.columns)  # guess allocations based on equal shares
    allocs_guess = np.ones(size) / size

    # Find symbols optimal allocation based on Sharpe ration
    bnds = tuple((0, 1) for x in range(size))  # duplicate tuple size times in a tuple
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    allocs_new = spo.minimize(getMaxSharpe, allocs_guess, args=(norm_prices,), method='SLSQP', bounds=bnds,
                              constraints=cons)

    prices_allcs = norm_prices * allocs_new.x
    row_val_df = prices_allcs.sum(axis=1)

    # get stats for the portfolio
    daily_return = row_val_df.pct_change()
    cr = (row_val_df[-1] - row_val_df[0]) / row_val_df[0]
    adr = daily_return[1:].mean()  # average daily returns
    sddr = daily_return[1:].std()
    sr = getSharpe(norm_prices, allocs_new.x)
    allocs = allocs_new.x


    """
    SPY
    """
    # dealing with missing values in the data file for SPY
    prices_SPY.fillna(method='ffill', inplace=True)
    prices_SPY.fillna(method='bfill', inplace=True)
    #port_val = prices_SPY  # add code here to compute daily portfolio values

    norm_prices_SPY = prices_SPY / prices_SPY.ix[0, :]  # normalized prices
    daily_return_SPY = norm_prices_SPY.pct_change()
    cr_SPY = (norm_prices_SPY[-1] - norm_prices_SPY[0]) / norm_prices_SPY[0]
    adr_SPY = daily_return_SPY[1:].mean()  # average daily returns
    sddr_SPY = daily_return_SPY[1:].std()
    sr_SPY = getSharpe(norm_prices, 1)






    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:  		  	   		  	  			  		 			     			  	 
        # add code to plot here
        df_temp = pd.concat(
            [row_val_df, prices_SPY], keys=["Portfolio", "SPY"], axis=1
        )
        df_temp['SPY'] = df_temp['SPY'] / df_temp.iloc[0]['SPY']
        df_temp.plot()
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Daily Portfolio Value and SPY')
        plt.savefig('images/plot.png')
        plt.close()

    return allocs, cr, adr, sddr, sr
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def test_code():  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    This function WILL NOT be called by the auto grader.  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    start_date = dt.datetime(2009, 1, 1)  		  	   		  	  			  		 			     			  	 
    end_date = dt.datetime(2010, 1, 1)  		  	   		  	  			  		 			     			  	 
    symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # Assess the portfolio  		  	   		  	  			  		 			     			  	 
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		  	  			  		 			     			  	 
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # Print statistics  		  	   		  	  			  		 			     			  	 
    print(f"Start Date: {start_date}")  		  	   		  	  			  		 			     			  	 
    print(f"End Date: {end_date}")  		  	   		  	  			  		 			     			  	 
    print(f"Symbols: {symbols}")  		  	   		  	  			  		 			     			  	 
    print(f"Allocations:{allocations}")  		  	   		  	  			  		 			     			  	 
    print(f"Sharpe Ratio: {sr}")  		  	   		  	  			  		 			     			  	 
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		  	  			  		 			     			  	 
    print(f"Average Daily Return: {adr}")  		  	   		  	  			  		 			     			  	 
    print(f"Cumulative Return: {cr}")  		  	   		  	  			  		 			     			  	 

if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    # This code WILL NOT be called by the auto grader  		  	   		  	  			  		 			     			  	 
    # Do not assume that it will be called  		  	   		  	  			  		 			     			  	 
    test_code()  		  	   		  	  			  		 			     			  	 
