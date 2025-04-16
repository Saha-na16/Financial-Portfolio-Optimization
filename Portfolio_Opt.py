import yfinance as yf
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

#User Inputs
tickers = ['BAC', 'TSLA', 'KO']
start = dt.datetime(2014, 1, 1)
end = dt.datetime(2020, 11, 20)
number_of_portfolios = 150
RF = 0

#Returns Calculations
returns = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=start, end=end)
    data[ticker] = data['Close'].pct_change()  # Changed from 'Adj Close' to 'Close'

    if returns.empty:
        returns = data[[ticker]]
    else:
        returns = returns.join(data[[ticker]], how='outer')

print(returns.head())

portfolio_returns = []
portfolio_risks = []
sharpe_ratios = []
portfolio_weights = []

for portfolio in range(number_of_portfolios):
    #Generate random portfolio weights
    weights = np.random.random_sample(len(tickers))
    weights = np.round((weights / np.sum(weights)),3)
    portfolio_weights.append(weights)

    #Calculate annualized return
    annualized_return = np.sum(returns.mean() * weights) * 252
    portfolio_returns.append(annualized_return)

    #Matrix covariance & Portfolio risk calculation
    matrix_covariance = returns.cov() * 252
    portfolio_variance = np.dot(weights.T, np.dot(matrix_covariance, weights))
    portfolio_standard_deviation = np.sqrt(portfolio_variance)
    portfolio_risks.append(portfolio_standard_deviation)

    #Sharpe ratio
    sharpe_ratio = (annualized_return - RF) / portfolio_standard_deviation
    sharpe_ratios.append(sharpe_ratio)

portfolio_returns = np.array(portfolio_returns)
portfolio_risks = np.array(portfolio_risks)
sharpe_ratios = np.array(sharpe_ratios)

portfolio_metrics = [portfolio_returns, portfolio_risks, sharpe_ratios, portfolio_weights]

portfolios_df = pd.DataFrame(portfolio_metrics).T
portfolios_df.columns = ['Return', 'Risk', 'Sharpe', 'Weights']
print(portfolios_df)

min_risk = portfolios_df.iloc[portfolios_df['Risk'].astype(float).idxmin()]
highest_return = portfolios_df.iloc[portfolios_df['Return'].astype(float).idxmax()]
highest_sharpe = portfolios_df.iloc[portfolios_df['Sharpe'].astype(float).idxmax()]

print('Lowest risk')
print(min_risk)
print(tickers)
print('')

print('Highest return')
print(highest_return)
print(tickers)
print('')

print('Highest sharpe ratio')
print(highest_sharpe)
print(tickers)
print('')

#Visualization
plt.figure(figsize = (10,5))
plt.scatter(portfolio_risks, portfolio_returns,
            c = portfolio_returns / portfolio_risks)
plt.title('Portfolio Optimaization', fontsize = 26)
plt.xlabel('Volatility', fontsize = 20)
plt.ylabel('Return', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.colorbar(label = 'Shape ratio')
plt.show()
