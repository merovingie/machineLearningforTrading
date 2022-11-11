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

def fit_allocs(prices, sharpe_func, allocs_guess, size):
    """Find the best allocation for a portfolio
    based-on the historical prices of the stocks in it.
    :param prices: prices of the portfolio
    :param sharpe_func: the function to minimize
    :return: allocs: allocation of the stocks in a portfolio
    """
    # Setting boundary to be [0, 1];
    # constraints to be sum(allocs) == 1


    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnds = ((0, 1),) * size # duplicate tuple size times in a tuple

    allocs = spo.minimize(sharpe_func, allocs_guess, args=(prices,), method='SLSQP', bounds=bnds, constraints=cons)

    return allocs.x

def get_sharpe(daily_return, sf = 0, rfr = 252):
    sharpe = (np.sqrt(sf) * (daily_return[1:] - rfr).mean() / ((daily_return[1:] - rfr).std()))
    return sharpe

def get_max_sharpe(prices, alloc_guess, size):
    norm_prices = prices / prices.ix[0, :]  # normalized prices
    size = len(prices.columns)  # guess allocations based on equal shares
    allocs_guess = np.ones(size) / size
    alloced = norm_prices * allocs_guess
    row_val_df = alloced.sum(axis=1)

    daily_return = row_val_df.pct_change()
    avg_daily_return = daily_return[1:].mean()
    neg_sharpe_ratio = -1 * get_sharpe(daily_return)
    return neg_sharpe_ratio

def sharpe(daily_returns, rfr=0, samplingRate=252):
    """Compute Sharpe ration for a portfolio
    Input:
       allocs: allocation of a portfolio, note: allocs.sum() should be one and allocs in [0,1]
       prices: prices of all stocks in a portfolio
       rfr: risk-free return
       K: K_daily = sqrt(252); K_annually = sqrt(1); K_monthly = sqrt(12)
    return:
      Sharpe_ratio
    """

    sharpe_ratio = np.sqrt(samplingRate) * (daily_returns - rfr).mean() / daily_returns.std()
    return sharpe_ratio

# this is the function that the minimizer will work on.
def neg_sharpe(allocs, prices, rfr=0, samplingRate=252):
    normed = prices / prices.iloc[0,:]
    alloced = normed * allocs
    port_val = alloced.sum(axis=1)
    daily_returns = compute_daily_returns(port_val)
    neg_sharpe_ratio = -1 * sharpe(daily_returns, rfr, samplingRate)
    return neg_sharpe_ratio

def compute_daily_returns(df):
    """Compute and return daily returns"""
    daily_returns = (df / df.shift(1)) - 1
    daily_returns = daily_returns[1:]
    return daily_returns

def get_portfolio_values(df, alloc):
	dfNew = df/df.iloc[0]
	dfNew = dfNew * alloc
	portfolio_values = dfNew.sum(axis = 1)
	return portfolio_values
'''------------------------------------------------------'''
'''------------------------------------------------------'''
'''------------------------------------------------------'''
''' Test pilot code'''


def get_daily_returns(portfolio_values):
	daily_returns = portfolio_values.copy()
	daily_returns[1:] = portfolio_values[1:]/(portfolio_values[:-1].values) - 1
	daily_returns = daily_returns[1:]
	return daily_returns

def get_cumm_returns(portfolio_values):
	return (portfolio_values[-1]/portfolio_values[0]) - 1

def get_avg_daily_returns(daily_returns):
	return daily_returns.mean()

def get_std_daily_returns(daily_returns):
	return daily_returns.std()

def get_sharpe_ratio(daily_returns):
	return 15.874*get_avg_daily_returns(daily_returns)/get_std_daily_returns(daily_returns)

def minimize_sharpe_ratio(alloc, df):
	# print "Came here"
	portfolio_values = get_portfolio_values(df, alloc, 1)
	daily_returns = get_daily_returns(portfolio_values)
	sharpe_ratio = get_sharpe_ratio(daily_returns)
	# print alloc, sharpe_ratio
	return -sharpe_ratio

def linear_constraint(x):
	return np.sum(x) - 1
'''------------------------------------------------------'''
'''------------------------------------------------------'''
'''------------------------------------------------------'''
  		  	   		  	  			  		 			     			  	 
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
    sf = 252.0
    rfr = 0.0


    # dealing with missing values in the data file.
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

    #calculate basic states before calling maximize function
    norm_prices = prices / prices.ix[0, :]  # normalized prices
    size = len(prices.columns)  # guess allocations based on equal shares
    allocs_guess = np.ones(size) / size
    alloced = norm_prices * allocs_guess
    row_val_df = alloced.sum(axis=1)

    get_cum_retrn = get_cumm_returns(row_val_df)
    cumm_return = (row_val_df[-1] - row_val_df[0]) / row_val_df[0]
    commul = row_val_df.pct_change()
    adr1 = commul[1:].mean()


    daily_return = row_val_df.pct_change()
    avg_daily_return = daily_return[1:].mean()
    sharpe = get_sharpe(daily_return)
    #sharpe = np.sqrt(sf)*(daily_return[1:]-rfr).mean()/((daily_return[1:]-rfr).std())
    bnds = tuple((0, 1) for x in range(noa))
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    results = spo.minimize(run_assess_portfolio, allocs, args=(prices,), method='SLSQP', bounds=bnds, constraints=cons)



    #new_port_value = get_portfolio_values(norm_prices, allocs_guess) #should remove no need
    # Find symbols optimal allocation based on Sharpe ration
    allocs = fit_allocs(prices, get_max_sharpe, allocs_guess, size)  # get allocations with minimizer
  		  	   		  	  			  		 			     			  	 
    # Get daily portfolio value  		  	   		  	  			  		 			     			  	 
    # port_val = prices_SPY  # add code here to compute daily portfolio values
    normed = prices / prices.iloc[0, :]
    alloced = normed * allocs
    port_val = alloced.sum(axis=1)  # portfolio
    daily_returns = compute_daily_returns(port_val)

    # get stats for the portfolio
    daily_returns = compute_daily_returns(port_val)
    cr = (port_val[-1] - port_val[0]) / port_val[0]
    adr = daily_returns.mean()  # average daily returns
    sddr = daily_returns.std()
    sr = sharpe(daily_returns)


    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:  		  	   		  	  			  		 			     			  	 
        # add code to plot here
        df_temp = pd.concat(
            [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1  		  	   		  	  			  		 			     			  	 
        )
        df_temp['SPY'] = df_temp['SPY'] / df_temp.iloc[0]['SPY']
        df_temp.plot()
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Daily Portfolio Value and SPY')
        plt.savefig('plot.png')
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
        sd=start_date, ed=end_date, syms=symbols, gen_plot=False  		  	   		  	  			  		 			     			  	 
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
