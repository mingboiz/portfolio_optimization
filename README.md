# Portfolio Optimization with Quadratic Transaction Costs

- `arxiv` is our final report.
- `market_impact_final` is our documented code for the implementation of the Market Impact Model in Frazzini et al (2018), to estimate the transaction cost of any arbitrary trade. 
- `code` is our backtesting using alphas from non-linear factor modelling on US equities using RNNs, which also includes implementation for the Optimized Market Impact portfolio and the Optimized Mean-Variance portfolio. 
- `pf_daily-final` and `pf_results-final` contains our backtest results.
- Macquarie Quant Alpha Model numbers are taken from Borghi & Giuliano (2020).


## Results

|                           | Mean-Variance | Market Impact | Alpha Model |
|---------------------------|---------------|---------------|-------------|
| Gross Return              |  14.3%        |  13.9%        | 6.2%        |
| Net Return                | -18.6%        | -18.9%        |             |
| Gross Information Ratio   | 0.797         | 0.802         | 0.870       |
| Net Information Ratio     | 0.794         | 0.793         |             |
| Avg. Turnover             | 2.47          | 2.47          |             |
| Avg. Turnover – Optimized | -1.35         | -1.37         | 1.39        |
| Max Drawdown              | 42.2%         | 42.9%         | 11.9%       |
| No. of Observations       | 11            | 11            | 95          |



![Results](https://github.com/mingboi95/portfolio_optimization/blob/main/Backtest-daily-no-trading-costs.png?raw=true)


## Credits
Our work could not have been possible without [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)
