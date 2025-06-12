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

        # Calculate moving averages (fixed the bug from original)
        signals['short_mavg'] = self.bars['Close'].ewm(span=self.short_window).mean()
        signals['long_mavg'] = self.bars['Close'].ewm(span=self.long_window).mean()

        # Generate signals: 1 when short MA > long MA, 0 otherwise
        signals['signal'] = np.where(
            signals['short_mavg'] > signals['long_mavg'], 
            1.0, 
            0.0
        )
        
        # Calculate position changes (1 = buy, -1 = sell, 0 = hold)
        signals['positions'] = signals['signal'].diff()
        return signals
    
class FullInvestmentPortfolio(Portfolio):
    """Portfolio that invests entire equity on buy signals and calculates
    returns based on stock performance rather than capital allocation.
    
    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0) for each symbol.
    initial_capital - The starting portfolio value.
    strategy - The strategy object (to get window parameters)."""
    
    def __init__(self, symbol, bars, signals, initial_capital, strategy=None):
        self.symbol = symbol
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.strategy = strategy
        self.portfolio_results = self.backtest_portfolio()

    def backtest_portfolio(self):
        """Calculate portfolio performance based on full investment strategy"""
        
        # Initialize portfolio tracking
        portfolio = pd.DataFrame(index=self.bars.index)
        portfolio['Price'] = self.bars['Close']
        portfolio['Signal'] = self.signals['signal']
        portfolio['Position_Change'] = self.signals['positions']
        
        # Calculate daily returns of the stock
        portfolio['Stock_Returns'] = self.bars['Close'].pct_change()
        
        # Calculate strategy returns
        # When signal = 1, we get full stock returns
        # When signal = 0, we get 0 returns (out of market)
        portfolio['Strategy_Returns'] = portfolio['Signal'].shift(1) * portfolio['Stock_Returns']
        
        # Calculate cumulative portfolio value
        # Start with initial capital and compound the returns
        portfolio['Portfolio_Value'] = self.initial_capital * (1 + portfolio['Strategy_Returns']).cumprod()
        
        # Handle the first row (no previous signal)
        portfolio.loc[portfolio.index[0], 'Strategy_Returns'] = 0
        portfolio.loc[portfolio.index[0], 'Portfolio_Value'] = self.initial_capital
        
        # Calculate portfolio returns
        portfolio['Portfolio_Returns'] = portfolio['Portfolio_Value'].pct_change()
        
        # Add buy/sell markers for plotting
        portfolio['Buy_Signals'] = np.where(portfolio['Position_Change'] == 1.0, 
                                          portfolio['Portfolio_Value'], np.nan)
        portfolio['Sell_Signals'] = np.where(portfolio['Position_Change'] == -1.0, 
                                           portfolio['Portfolio_Value'], np.nan)
        
        return portfolio

    def calculate_cagr(self):
        """Calculate Compound Annual Growth Rate"""
        start_value = self.initial_capital
        end_value = self.portfolio_results['Portfolio_Value'].iloc[-1]
        
        # Calculate number of years
        start_date = self.portfolio_results.index[0]
        end_date = self.portfolio_results.index[-1]
        years = (end_date - start_date).days / 365.25
        
        # CAGR formula: (End Value / Start Value)^(1/years) - 1
        cagr = (end_value / start_value) ** (1 / years) - 1
        
        return cagr, start_value, end_value, years

    def get_performance_stats(self):
        """Get key performance statistics"""
        cagr, start_value, end_value, years = self.calculate_cagr()
        
        total_return = (end_value - start_value) / start_value
        
        # Calculate volatility (annualized)
        daily_returns = self.portfolio_results['Strategy_Returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # 252 trading days per year
        
        # Calculate maximum drawdown
        portfolio_values = self.portfolio_results['Portfolio_Value']
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            'CAGR': cagr,
            'Total Return': total_return,
            'Volatility': volatility,
            'Max Drawdown': max_drawdown,
            'Start Value': start_value,
            'End Value': end_value,
            'Years': years
        }

def create_buy_hold_portfolio(symbol, bars, initial_capital):
    """Create a buy and hold portfolio for comparison"""
    portfolio = pd.DataFrame(index=bars.index)
    portfolio['Price'] = bars['Close']
    
    # Calculate daily returns
    portfolio['Stock_Returns'] = bars['Close'].pct_change()
    
    # Buy and hold: always invested (signal = 1)
    portfolio['Strategy_Returns'] = portfolio['Stock_Returns']
    
    # Calculate cumulative portfolio value
    portfolio['Portfolio_Value'] = initial_capital * (1 + portfolio['Strategy_Returns']).cumprod()
    
    # Handle the first row
    portfolio.loc[portfolio.index[0], 'Strategy_Returns'] = 0
    portfolio.loc[portfolio.index[0], 'Portfolio_Value'] = initial_capital
    
    return portfolio

def calculate_buy_hold_stats(portfolio_data, initial_capital):
    """Calculate performance statistics for buy and hold"""
    start_value = initial_capital
    end_value = portfolio_data['Portfolio_Value'].iloc[-1]
    
    # Calculate number of years
    start_date = portfolio_data.index[0]
    end_date = portfolio_data.index[-1]
    years = (end_date - start_date).days / 365.25
    
    # CAGR formula
    cagr = (end_value / start_value) ** (1 / years) - 1
    total_return = (end_value - start_value) / start_value
    
    # Calculate volatility (annualized)
    daily_returns = portfolio_data['Strategy_Returns'].dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    
    # Calculate maximum drawdown
    portfolio_values = portfolio_data['Portfolio_Value']
    peak = portfolio_values.expanding().max()
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        'CAGR': cagr,
        'Total Return': total_return,
        'Volatility': volatility,
        'Max Drawdown': max_drawdown,
        'Start Value': start_value,
        'End Value': end_value,
        'Years': years
    }

def plot_results(portfolio_obj, symbol, bars, initial_capital):
    """Plot the strategy results with comparisons to buy & hold"""
    
    portfolio_data = portfolio_obj.portfolio_results
    strategy_stats = portfolio_obj.get_performance_stats()
    
    # Create buy and hold portfolios for comparison
    stock_buy_hold = create_buy_hold_portfolio(symbol, bars, initial_capital)
    stock_bh_stats = calculate_buy_hold_stats(stock_buy_hold, initial_capital)
    
    # Download S&P 500 data for the same period
    print("Downloading S&P 500 data for comparison...")
    sp500_data = yf.download('^GSPC', start=bars.index[0], end=bars.index[-1], auto_adjust=True)
    sp500_buy_hold = create_buy_hold_portfolio('^GSPC', sp500_data, initial_capital)
    sp500_bh_stats = calculate_buy_hold_stats(sp500_buy_hold, initial_capital)
    
    # Create the plot with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    
    # Top Left: Stock price with moving averages and signals
    ax1.plot(portfolio_data.index, portfolio_data['Price'], 
             color='blue', lw=1.5, label=f'{symbol} Price')
    
    # Plot moving averages
    signals = portfolio_obj.signals
    short_window = portfolio_obj.strategy.short_window if portfolio_obj.strategy else 50
    long_window = portfolio_obj.strategy.long_window if portfolio_obj.strategy else 200
    
    ax1.plot(signals.index, signals['short_mavg'], 
             color='red', lw=1, label=f'MA{short_window}')
    ax1.plot(signals.index, signals['long_mavg'], 
             color='green', lw=1, label=f'MA{long_window}')
    
    # Plot buy/sell signals on price chart
    buy_signals = portfolio_data['Position_Change'] == 1.0
    sell_signals = portfolio_data['Position_Change'] == -1.0
    
    ax1.scatter(portfolio_data.index[buy_signals], 
               portfolio_data['Price'][buy_signals],
               marker='^', s=100, color='green', label='Buy Signal', zorder=5)
    ax1.scatter(portfolio_data.index[sell_signals], 
               portfolio_data['Price'][sell_signals],
               marker='v', s=100, color='red', label='Sell Signal', zorder=5)
    
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{symbol} Price with MA Crossover Strategy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top Right: Strategy vs Stock Buy & Hold Comparison
    ax2.plot(portfolio_data.index, portfolio_data['Portfolio_Value'], 
             color='purple', lw=2, label='MA Cross Strategy')
    ax2.plot(stock_buy_hold.index, stock_buy_hold['Portfolio_Value'], 
             color='blue', lw=2, label=f'{symbol} Buy & Hold')
    
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.set_title(f'Strategy vs {symbol} Buy & Hold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add strategy vs stock comparison stats
    strategy_text = f"""MA Cross Strategy:
CAGR: {strategy_stats['CAGR']:.2%}
Total Return: {strategy_stats['Total Return']:.2%}
Max Drawdown: {strategy_stats['Max Drawdown']:.2%}

{symbol} Buy & Hold:
CAGR: {stock_bh_stats['CAGR']:.2%}
Total Return: {stock_bh_stats['Total Return']:.2%}
Max Drawdown: {stock_bh_stats['Max Drawdown']:.2%}"""
    
    ax2.text(0.02, 0.98, strategy_text, transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
             verticalalignment='top', fontsize=9, fontfamily='monospace')
    
    # Bottom Left: Strategy vs S&P 500 Comparison
    ax3.plot(portfolio_data.index, portfolio_data['Portfolio_Value'], 
             color='purple', lw=2, label='MA Cross Strategy')
    ax3.plot(sp500_buy_hold.index, sp500_buy_hold['Portfolio_Value'], 
             color='orange', lw=2, label='S&P 500 Buy & Hold')
    
    ax3.set_ylabel('Portfolio Value ($)')
    ax3.set_xlabel('Date')
    ax3.set_title('Strategy vs S&P 500 Buy & Hold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add strategy vs S&P 500 comparison stats
    sp500_text = f"""MA Cross Strategy:
CAGR: {strategy_stats['CAGR']:.2%}
Volatility: {strategy_stats['Volatility']:.2%}

S&P 500 Buy & Hold:
CAGR: {sp500_bh_stats['CAGR']:.2%}
Volatility: {sp500_bh_stats['Volatility']:.2%}

Alpha: {strategy_stats['CAGR'] - sp500_bh_stats['CAGR']:.2%}"""
    
    ax3.text(0.02, 0.98, sp500_text, transform=ax3.transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
             verticalalignment='top', fontsize=9, fontfamily='monospace')
    
    # Bottom Right: All Three Strategies Normalized Comparison
    # Normalize all portfolios to start at 100 for easy comparison
    strategy_normalized = (portfolio_data['Portfolio_Value'] / initial_capital) * 100
    stock_bh_normalized = (stock_buy_hold['Portfolio_Value'] / initial_capital) * 100
    sp500_bh_normalized = (sp500_buy_hold['Portfolio_Value'] / initial_capital) * 100
    
    ax4.plot(portfolio_data.index, strategy_normalized, 
             color='purple', lw=2, label='MA Cross Strategy')
    ax4.plot(stock_buy_hold.index, stock_bh_normalized, 
             color='blue', lw=2, label=f'{symbol} Buy & Hold')
    ax4.plot(sp500_buy_hold.index, sp500_bh_normalized, 
             color='orange', lw=2, label='S&P 500 Buy & Hold')
    
    ax4.set_ylabel('Normalized Value (Base = 100)')
    ax4.set_xlabel('Date')
    ax4.set_title('Normalized Performance Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add final values
    final_text = f"""Final Values (Normalized):
MA Cross: {strategy_normalized.iloc[-1]:.1f}
{symbol} B&H: {stock_bh_normalized.iloc[-1]:.1f}
S&P 500 B&H: {sp500_bh_normalized.iloc[-1]:.1f}"""
    
    ax4.text(0.02, 0.98, final_text, transform=ax4.transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8),
             verticalalignment='top', fontsize=9, fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    return strategy_stats, stock_bh_stats, sp500_bh_stats

if __name__ == "__main__":
    # Set up the strategy parameters
    symbol = 'TSLA'
    short_window = 100
    long_window = 400
    initial_capital = 1.0
    
    print(f"Downloading {symbol} data...")
    # Download stock data
    bars = yf.download(symbol, start='2000-01-01', 
                      end=datetime.today().strftime('%Y-%m-%d'), 
                      auto_adjust=True)
    
    print(f"Running Moving Average Crossover Strategy...")
    print(f"Short MA: {short_window} days, Long MA: {long_window} days")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    
    # Create strategy and generate signals
    strategy = MovingAverageCrossStrategy(symbol, bars, short_window, long_window)
    signals = strategy.generate_signals()
    
    # Create portfolio and run backtest
    portfolio = FullInvestmentPortfolio(symbol, bars, signals, initial_capital, strategy)
    
    # Plot results and display statistics
    strategy_stats, stock_bh_stats, sp500_bh_stats = plot_results(portfolio, symbol, bars, initial_capital)