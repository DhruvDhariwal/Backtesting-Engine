# Backtesting-Engine

*backtest.py* 
Contains the abstract classes that provides the structure for every strategy made along with how the portfolio is managed.

*ma_cross.py* 
First iteration of the moving average strategy. It works by just buying a pre-set amount of shares when the golden cross signal is achieved. Plots the stock data along with the moving averages in the given time frame and also the portfolio total.

*ma_cross2.py* 
Instead of buying a set amount of shares each time, this works by calculating the returns achieved by the strategy. Provides more statistics like CAGR and volatility and drawdown

*ma_cross3.py*
Builds upon *ma_cross2.py* by comparing the strategy with the S&P500 returns and also with a buy and hold strategy for the same stock in the given time period. More detailed graphs and statistiics are provided.
