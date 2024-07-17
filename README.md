# Index-Weighting-Strategies

This repository contains a collection of functions and classes designed to explore and implement various index weighting strategies. The primary focus is on financial indices and how different weighting methods can be applied to them. The code is written in Python and leverages several powerful libraries for data manipulation, statistical analysis, and visualization.

## Features

- **Ken French Industry Portfolios**: A class to load and format industry portfolio data
- **Wavelet Transforms**: Utilize the PyWavelets library for advanced data transformation
- **Clustering and PCA**: Implement clustering algorithms and Principal Component Analysis (PCA) for dimensionality reduction
- **Optimization**: Use the SciPy library to perform portfolio optimization
- **Visualization**: Create detailed and interactive plots using Matplotlib and Plotly

## Repository Structure

The repository is organized as follows:

- `data/`: Contains data files related to Ken French Industry Portfolios
- `Index_weighting_functions.py`: Python file containing various index weighting functions and methods
- `LICENSE`: The license for the project
- `README.md`: This README file
- `ind49_weighting_strategies.ipynb`: Jupyter notebook demonstrating the use of different weighting strategies

## Requirements

The code requires the following Python libraries:

- numpy
- pandas
- matplotlib
- plotly
- scipy
- scikit-learn
- PyWavelets

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib plotly scipy scikit-learn PyWavelets
```

## Weighting Methods

The repository explores several different methods for weighting indices. Here are the main methods utilized:

1. **Equal Weighting (EW):**
   - Each asset in the index is given an equal weight, regardless of its market capitalization or other metrics. This method is straightforward and ensures that each asset has the same impact on the index’s performance. Cap weights and thresholds can be applied to adjust the basic equal weighting strategy
2. **Value Weighting (VW):**
   - Assets are weighted according to their market value, which is the product of the asset’s price and the number of shares outstanding. This method favors larger companies, reflecting their market dominance and liquidity
3. **Global Minimum Variance (GMV):**
   - The weights are optimized to minimize the overall variance of the portfolio, considering the covariance between asset returns. This approach aims to construct a portfolio with the lowest possible volatility
4. **Equal Risk Contribution (ERC):**
   - Weights are assigned to ensure that each asset contributes equally to the overall portfolio risk. This method balances the risk contribution of each asset, leading to a more stable and diversified portfolio
5. **Minimum Correlation:**
   - This method aims to minimize the sum of all pairwise correlations in the portfolio. The goal is to create a portfolio with assets that have low correlations with each other, enhancing diversification
6. **Maximum Diversification:**
   - This method seeks to maximize the diversification ratio, which is the ratio of the weighted sum of individual asset volatilities to the portfolio volatility. The goal is to enhance the diversification benefits by spreading risk more effectively across assets
7. **Momentum:**
   - Portfolio weights are based on cross-sectional momentum, where assets with higher past returns receive higher weights. The lookback period is used to calculate momentum
8. **K-means Clustering:**
   - Portfolio weights are generated using K-means clustering on asset returns. Assets are grouped into clusters, and weights are assigned based on cluster membership
9. **Hierarchical Clustering:**
   - This method uses hierarchical clustering on asset returns to group assets into clusters with similar behavior. Weights are assigned within and between clusters to minimize risk
10. **Hierarchical Risk Parity (HRP):**
    - A sophisticated method that uses hierarchical clustering to build a diversified portfolio. HRP groups assets into clusters with similar behavior and then allocates weights within and between clusters to minimize risk
11. **Principal Component Analysis (PCA):**
    - Portfolio weights are generated using PCA on asset returns. The first principal component is used to determine the weights, capturing the most significant variance in the data
12. **Tail Risk Parity (TRP):**
    - This approach generates portfolio weights based on the Tail Risk Parity method, which considers the Conditional Value at Risk (CVaR) to balance tail risks across assets
13. **Wavelet:**
    - Weights are calculated based on the wavelet variance of each asset’s returns. This method uses wavelet transforms to capture variations at different frequencies
14. **Maximum Sharpe Ratio (MSR):**
    - This method optimizes the portfolio by maximizing the Sharpe ratio using historical return data. The Sharpe ratio is a measure of risk-adjusted return
15. **Algorithmic Complexity:**
    - Portfolio weights are based on the Lempel-Ziv complexity of each asset’s returns, with higher complexity assets receiving higher weights
16. **Random Forest (RF):**
    - Portfolio weights are generated using a Random Forest model based on historical return data. This method uses machine learning to determine the importance of each asset

## Performance Analysis

To analyze the performance of different weighting strategies, we use a summary statistics table that includes various performance and risk metrics. The key metrics considered are numerous, but only a few will be included here:

- **Sharpe Ratio:** A measure of risk-adjusted return, calculated as the ratio of excess return to standard deviation
- **Value at Risk (VaR):** A statistical technique to measure the risk of loss for investments
- **Net Profit to Worst Drawdown:** The ratio of net profit to the maximum drawdown
- **Calmar Ratio:** The ratio of annualized return to the maximum drawdown

These metrics provide a comprehensive view of the portfolio’s performance, helping to evaluate the effectiveness of different weighting strategies.

## License

This project is licensed under the MIT License.


---

This README provides an overview of the Index-Weighting-Strategies repository, including its features, requirements, usage, and detailed descriptions of various weighting methods and performance analysis metrics.
