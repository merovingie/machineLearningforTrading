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

'''
import datetime as dt  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import os  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

import numpy as np
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import pandas as pd  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
from util import get_data, plot_data  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

def author():
    return 'aladdha7'

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

    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    orders.sort_index(inplace=True)

    start_date = orders.index.min()
    end_date = orders.index.max()
    stocks = list(orders.Symbol.unique())

   #Getting Stock Prices for the range start_date - end_date
    prices = get_data(stocks, pd.date_range(start_date, end_date))
    prices.ffill(inplace=True)    #Forward Filling first
    prices.bfill(inplace=True)    #Backward filling

    #SPY gets automatically added using get_data. So, delete SPY if not in the list of orders
    if 'SPY' not in stocks:
        prices.drop('SPY', axis=1, inplace=True)

    prices['Cash'] = 1   #Adding a cash field to prices and initializing to 1 so that we can multiply it straight away with holdings

    trade = pd.DataFrame(np.zeros(prices.shape), columns=prices.columns, index=prices.index)
    trade.iloc[0, -1] = start_val  #Cash for first date set to start value

    for index, row in orders.iterrows():
        stock = row['Symbol']
        order_type = row['Order']
        shares = row['Shares']
        stock_price = prices.loc[index, stock]  # getting stock price from prices df

        if order_type == 'SELL':
            multiplier = -1
            trade.loc[index, stock] = trade.loc[index, stock] - shares
            stock_price = stock_price - (stock_price * impact)

        else:
            multiplier = 1
            trade.loc[index, stock] = trade.loc[index, stock] + shares
            stock_price = stock_price + (stock_price * impact)

        # accounting market impact
        trade.loc[index, 'Cash'] = trade.loc[index, 'Cash'] - commission - (stock_price * shares * multiplier)

    holding = trade.cumsum()
    holding_value = holding * prices  # computing stock total values
    portvals = holding_value.sum(axis=1)

    return portvals


def Portfolio_Statistics(portvals):
    daily_returns = portvals.copy()
    daily_returns[1:] = (portvals[1:] / portvals[:-1].values) - 1
    daily_returns = daily_returns[1:]
    # print(daily_returns)

    cum_ret = (portvals[-1] / portvals[0]) - 1
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    sharpe_ratio = np.sqrt(252.0) * (avg_daily_ret / std_daily_ret)

    return cum_ret,avg_daily_ret,std_daily_ret,sharpe_ratio


def SPY_Statistics(start_date, end_date):

    SPY_prices = get_data(['$SPX'], pd.date_range(start_date, end_date))
    SPY_prices.drop('SPY', axis=1, inplace=True)

    daily_returns = SPY_prices.copy()
    daily_returns[1:] = (SPY_prices[1:] / SPY_prices[:-1].values) - 1
    daily_returns = daily_returns[1:]
    # print(daily_returns)

    cum_ret = (SPY_prices.iloc[-1] / SPY_prices.iloc[0]) - 1
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

    of = "./orders/orders.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[
            0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
    #     0.2,
    #     0.01,
    #     0.02,
    #     1.5,
    # ]
    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
    #     0.2,
    #     0.01,
    #     0.02,
    #     1.5,
    # ]

    orders_temp = pd.read_csv(of, index_col='Date', parse_dates=True, na_values=['nan'])
    start_date = orders_temp.index.min()
    end_date = orders_temp.index.max()

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = Portfolio_Statistics(portvals = portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY =  SPY_Statistics(start_date = start_date, end_date =end_date)

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
'''
'''
"""MC2-P1: Market simulator.
orders.csv: https://youtu.be/1ysZptg2Ypk?t=3m48s
df_prices: https://youtu.be/1ysZptg2Ypk?t=6m30s
df_trades: https://youtu.be/1ysZptg2Ypk?t=9m9s
df_holdings: https://youtu.be/1ysZptg2Ypk?t=15m47s
"""

import pandas as pd
import os
os.chdir('../')
from util import get_data, plot_data
os.chdir('./Project_5')

def compute_portvals(orders_file = "./data/", start_val = 1000000, commission=9.95, impact=0.005):
    """Returns a single column df with porfolio values from the beginning to the end of the order file"""
    # orders columns: Date, symbol, order, shares
    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True) 
    orders = orders.sort_index()

    symbols = list(set(orders['Symbol'])) 
    dates = list(set(orders.index)) # dates where things actually happen
    dates.sort()
    
    # prices data is the Adj close price per trading day
    prices_data = get_data(symbols, pd.date_range(dates[0],dates[-1]))
    if symbols.count('SPY') == 0: # SPY is kept to maintain trading days, removed if not part of portfolio, get_data adds it automatically
        prices_data = prices_data.drop('SPY', axis=1)
        
    # df_prices columns: Date, *symbols, cash
    # df_prices is simply price data as a dataframe
    df_prices = pd.DataFrame(prices_data)
    df_prices['cash'] = 1
    
    # df_trades represents number of shares held and cash avalable only on order dates
    df_trades = df_prices.copy()
    df_trades[:] = 0
    
    # df_holdings represents df_trades, but on every date between sd and ed
    df_holdings = df_trades.copy()    
    
    date_commission = dict([[d, 0] for d in dates]) # keeps track of comission and market impact per date as key
    for index, col in orders.iterrows():
        if col['Order'] == 'SELL':    
            df_trades.loc[index, col['Symbol']] += col['Shares'] * -1 
        else:
            df_trades.loc[index, col['Symbol']] += col['Shares'] 
            
        # used orders as opposed to df_trades because same day trades may result in zero shares for the particular day
        date_commission[index] -= commission + (col['Shares'] * df_prices.loc[index, col['Symbol']] * impact) 
        
    for index in dates:
        df_trades.loc[index, 'cash'] += -1*(df_trades.ix[index, :-1].multiply(df_prices.ix[index, :-1]).sum())\
                                         + date_commission[index] 
                                         
    df_holdings.ix[0,'cash'] = start_val + df_trades.ix[0,'cash']
    df_holdings.iloc[0, :-1] = df_trades.iloc[0, :-1]
    
    for i in range(1, df_holdings.shape[0]):
        df_holdings.iloc[i, :] = df_trades.iloc[i, :] + df_holdings.iloc[i-1, :]
    print('DF_HOLDINGS')
    print(df_holdings)
    print('DF TRADES')
    print(df_trades)
    # df_value is the value of each holding and total as a dollar amount
    df_value = df_holdings.multiply(df_prices)
    df_portval = df_value.sum(axis=1)
    return df_portval
    
def author():
    return('mtong31')
    
def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "../../data/orders/orders-10.csv" #1026658.3265
#    of = "./orders/orders-11.csv"
#    of = "./orders/orders-12.csv" #1705686.6665
    sv = 1000000

    # Process orders
    if of == "../../data/orders/orders-12.csv":
        port_value = compute_portvals(orders_file = of, start_val = sv, commission=0)
    else:
        port_value = compute_portvals(orders_file = of, start_val = sv, commission=9.95)
        
    if isinstance(port_value, pd.DataFrame):
        port_value = port_value[port_value.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    d_returns       = port_value.copy() 
    d_returns       = (port_value[1:]/port_value.shift(1) - 1)
    d_returns.ix[0] = 0
    d_returns       = d_returns[1:]
    
    #Below are desired output values
    
    #Cumulative return (final - initial) - 1
    cr   = port_value[-1] / port_value[0] - 1
    #Average daily return
    adr  = d_returns.mean()
    #Standard deviation of daily return
    sddr = d_returns.std()
    #Sharpe ratio ((Mean - Risk free rate)/Std_dev)
    daily_rfr     = (1.0)**(1/252) - 1 #Should this be sampling freq instead of 252? 
    sr            = (d_returns - daily_rfr).mean() / sddr
    sr_annualized = sr * (252**0.5)


    # Compare portfolio against $SPX
    print("Date Range: {} to {}".format(port_value.index[0], port_value.index[-1],end='\n'))
    print("Sharpe Ratio of Fund: {}".format(sr_annualized))
#    print("Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY,end='\n'))
    print("Cumulative Return of Fund: {}".format(cr))
#    print("Cumulative Return of SPY : {}".format(cum_ret_SPY,end='\n'))
    print("Standard Deviation of Fund: {}".format(sddr))
#    print("Standard Deviation of SPY : {}".format(std_daily_ret_SPY,end='\n'))
    print("Average Daily Return of Fund: {}".format(adr))
#    print("Average Daily Return of SPY : {}".format(avg_daily_ret_SPY,end='\n'))
    print("\nFinal Portfolio Value: {}\n".format(port_value[-1]))
    
    return(compute_portvals(orders_file = of, start_val = sv))
if __name__ == "__main__":
   s = test_code()
'''