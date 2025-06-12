from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from backtest import Strategy, Portfolio

class MovingAverageCrossStrategy(Strategy):
    """    
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average."""

    def __init__(self, symbol, bars, short_window, long_window):
        self.symbol = symbol
        self.bars = bars
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0

        signals['short_mavg'] = self.bars['Close'].ewm(span = self.short_window).mean()
        signals['long_mavg'] = self.bars['Close'].ewm(span = self.long_window).mean()

        signals['signal'] = np.where(signals['short_mavg'] > signals['long_mavg'], 1.0, 0.0)
        
        signals['positions'] = signals['signal'].diff()
        return signals
    
class MarketOnClosePortfolio(Portfolio):
    """Encapsulates the notion of a portfolio of positions based
    on a set of signals as provided by a Strategy.

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio."""
    
    def __init__(self, symbol, bars, signals, initial_capital):
        self.symbol = symbol
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()

    def generate_positions(self):
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.symbol] = 500*self.signals['signal']
        return positions
    
    def backtest_portfolio(self):
        portfolio = self.positions*self.bars['Close']
        pos_diff = self.positions.diff()

        portfolio['Holdings'] = (self.positions*self.bars['Close']).sum(axis=1)
        portfolio['Cash'] = self.initial_capital - (pos_diff*self.bars['Close']).sum(axis=1).cumsum()

        portfolio['Total'] = portfolio['Cash'] + portfolio['Holdings']
        portfolio['Returns'] = portfolio['Total'].pct_change()

        return portfolio
    

if __name__ == "__main__":

    symbol = 'AAPL'
    bars = yf.download(symbol, start='1990-01-01', end=datetime.today().strftime('%Y-%m-%d') , auto_adjust=True)

    mac = MovingAverageCrossStrategy(symbol, bars, short_window=100, long_window=400)
    signals = mac.generate_signals()

    portfolio = MarketOnClosePortfolio(symbol, bars, signals, initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(211, ylabel='Price in $')
    bars['Close'].plot(ax=ax1, color='blue', lw=1.5)
    signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=1)

    ax1.plot(signals.loc[signals.positions == 1.0].index, 
             signals.short_mavg[signals.positions == 1.0],
             '^', markersize=7, color='g')
    
    ax1.plot(signals.loc[signals.positions == -1.0].index, 
             signals.short_mavg[signals.positions == -1.0],
             'v', markersize=7, color='r')
    
    ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
    returns['Total'].plot(ax=ax2, lw=1.5)

    ax2.plot(returns.loc[signals.positions == 1.0].index, 
             returns.Total[signals.positions == 1.0],
             '^', markersize=7, color='g')
    ax2.plot(returns.loc[signals.positions == -1.0].index, 
             returns.Total[signals.positions == -1.0],
             'v', markersize=7, color='r')
    
    plt.show()
    

