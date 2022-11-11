import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data
import indicators
import TheoreticallyOptimalStrategy as tos

def author():
    return 'rmikhael3'


if __name__ == "__main__":
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    symbol = "JPM"

    # Indicators
    indicators.graphIndicators(symbol, sd, ed)

    # TOS
    df_trades = tos.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=100000)

    # Stratgy bench marking
    trades_benchmark = tos.benchmarkTOS(symbol="JPM", sd=sd, ed=ed, shares=1000)

    # TOS vs Benchamrk Strategy
    tos.tosVsBenchmark(df_trades, trades_benchmark, symbol=symbol, sd=sd, ed=ed)