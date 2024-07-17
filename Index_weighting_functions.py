import pywt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from scipy.stats import norm
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor


class KenFrenchIndustryPortfolios:
    def __init__(self, n_inds=30):
        self.n_inds = n_inds

    def _get_ind_file(self, filetype, weighting="vw"):
        """
        Load and format the Ken French Industry Portfolios files
        Variant is a tuple of (weighting, size) where:
            weighting is one of "ew", "vw"
            number of inds is 30 or 49
        """    
        if filetype == "returns":
            name = f"{weighting}_rets" 
            divisor = 100
        elif filetype == "nfirms":
            name = "nfirms"
            divisor = 1
        elif filetype == "size":
            name = "size"
            divisor = 1
        else:
            raise ValueError(f"filetype must be one of: returns, nfirms, size")
        ind = pd.read_csv(f"data/ind{self.n_inds}_m_{name}.csv", header=0, index_col=0, na_values=-99.99)/divisor
        ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
        ind.columns = ind.columns.str.strip()
        return ind

    def get_ind_returns(self, weighting="vw"):
        """
        Load and format the Ken French Industry Portfolios Monthly Returns
        """
        return self._get_ind_file("returns", weighting=weighting)

    def get_ind_nfirms(self):
        """
        Load and format the Ken French 30 Industry Portfolios Average number of Firms
        """
        return self._get_ind_file("nfirms")

    def get_ind_size(self):
        """
        Load and format the Ken French 30 Industry Portfolios Average size (market cap)
        """
        return self._get_ind_file("size")

    def get_ind_market_caps(self, weights=False):
        """
        Load the industry portfolio data and derive the market caps
        """
        ind_nfirms = self.get_ind_nfirms()
        ind_size = self.get_ind_size()
        ind_mktcap = ind_nfirms * ind_size
        if weights:
            total_mktcap = ind_mktcap.sum(axis=1)
            ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
            return ind_capweight
        #else
        return ind_mktcap


class PortfolioStrategy:
    def __init__(self, data, lookback_years=5):
        self.data = data
        self.lookback_years = lookback_years
        self.weights_df = pd.DataFrame()
        self.btr_df = pd.DataFrame()

    def create_strategy(self, strategies, **kwargs):
        """
        Generate DataFrames for portfolio weights and backtest results directly
        """
        weights_dict = {}
        btr_dict = {}
        for strategy_name, strategy_func in strategies.items():
            btr_results = self.backtest(weight_func=lambda x: strategy_func(x, **kwargs), **kwargs)
            btr_dict[strategy_name] = btr_results['out_sample_return'].tolist()
            weights_dict[strategy_name] = btr_results['out_sample_weights']
        self.weights_df = pd.concat(weights_dict, axis=1)
        self.btr_df = pd.DataFrame(btr_dict)
        return self.weights_df, self.btr_df

    def add_strategy(self, strategy_name, strategy_func, **kwargs):
        """
        Add a new strategy and update the existing DataFrames
        """
        btr_results = self.backtest(weight_func=lambda x: strategy_func(x, **kwargs), **kwargs)
        self.btr_df[strategy_name] = btr_results['out_sample_return'].tolist()
        self.weights_df[strategy_name] = btr_results['out_sample_weights']
        return self.weights_df, self.btr_df
    
    def backtest(self, weight_func, **kwargs):
        """
        Perform a rolling backtest on the provided data with specified in-sample and out-sample periods
        """
        lookback_months = self.lookback_years * 12
        out_sample_months = 1
        results = pd.DataFrame(columns=['out_sample_return', 'out_sample_weights'])
        for i in range(lookback_months, len(self.data), out_sample_months):
            out_sample_start = i
            out_sample_end = out_sample_start + out_sample_months
            if out_sample_end > len(self.data):
                break
            in_sample_data = self.data.iloc[out_sample_start-lookback_months:out_sample_start]
            out_sample_weights = weight_func(in_sample_data, **kwargs)
            out_sample_weights = pd.Series(out_sample_weights, index=self.data.columns)
            out_sample_data = self.data.iloc[out_sample_start:out_sample_end]
            out_sample_returns = (out_sample_data * out_sample_weights).sum(axis=1)
            if not out_sample_returns.empty:
                scalar_return = out_sample_returns.iloc[0]
            results.loc[self.data.index[out_sample_start]] = {
                'out_sample_return': scalar_return,
                'out_sample_weights': out_sample_weights.to_dict()
            }
        return results
    

class FinancialMetrics:
    @staticmethod
    def _annualize(metric_func, r, periods_per_year, **kwargs):
        result = r.aggregate(metric_func, periods_per_year=periods_per_year, **kwargs)
        return result

    @staticmethod
    def _risk_free_adjusted_returns(r, riskfree_rate, periods_per_year):
        rf_per_period = (1 + riskfree_rate)**(1 / periods_per_year) - 1
        return r - rf_per_period

    @staticmethod
    def drawdown(return_series: pd.Series):
        """
        Takes a time series of asset returns
        Computes and returns a data frame that contains:
        the wealth index, the previous peaks, and percent drawdowns
        """
        wealth_index = 1000 * (1 + return_series).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdown = (wealth_index - previous_peaks) / previous_peaks
        return pd.DataFrame({
            "Wealth": wealth_index,
            "Peaks": previous_peaks,
            "Drawdown": drawdown
            })

    @staticmethod
    def semideviation(r, periods_per_year):
        """
        Compute the Annualized Semi-Deviation
        """
        neg_rets = r[r < 0]
        return FinancialMetrics.annualize_vol(r=neg_rets, periods_per_year=periods_per_year)

    @staticmethod
    def skewness(r):
        """
        Computes the skewness of the supplied Series or DataFrame
        Returns a float or a Series
        """
        demeaned_r = r - r.mean()
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**3).mean()
        return exp / sigma_r**3

    @staticmethod
    def kurtosis(r):
        """
        Computes the kurtosis of the supplied Series or DataFrame
        Returns a float or a Series
        """
        demeaned_r = r - r.mean()
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**4).mean()
        return exp / sigma_r**4

    @staticmethod
    def var_historic(r, level=5):
        """
        VaR Historic
        """
        if isinstance(r, pd.DataFrame):
            return r.aggregate(FinancialMetrics.var_historic, level=level)
        elif isinstance(r, pd.Series):
            return -np.percentile(r, level)
        else:
            raise TypeError("Expected r to be Series or DataFrame")

    @staticmethod
    def var_gaussian(r, level=5, modified=False):
        """
        Returns the Parametric Gaussian VaR of a Series or DataFrame
        If "modified" is True, then the modified VaR is returned,
        using the Cornish-Fisher modification
        """
        z = norm.ppf(level / 100)
        if modified:
            s = FinancialMetrics.skewness(r)
            k = FinancialMetrics.kurtosis(r)
            z = (z +
                 (z**2 - 1) * s / 6 +
                 (z**3 - 3 * z) * (k - 3) / 24 -
                 (2 * z**3 - 5 * z) * (s**2) / 36)
        return -(r.mean() + z * r.std(ddof=0))

    @staticmethod
    def cvar_historic(r, level=5):
        """
        Computes the Conditional VaR of Series or DataFrame
        """
        if isinstance(r, pd.Series):
            is_beyond = r <= -FinancialMetrics.var_historic(r, level=level)
            return -r[is_beyond].mean()
        elif isinstance(r, pd.DataFrame):
            return r.aggregate(FinancialMetrics.cvar_historic, level=level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")

    @staticmethod
    def annualize_rets(r, periods_per_year):
        """
        Annualizes a set of returns
        """
        compounded_growth = (1 + r).prod()
        n_periods = r.shape[0]
        return compounded_growth**(periods_per_year / n_periods) - 1

    @staticmethod
    def annualize_vol(r, periods_per_year):
        """
        Annualizes the vol of a set of returns
        """
        return r.std() * (periods_per_year**0.5)

    @staticmethod
    def sharpe_ratio(r, riskfree_rate, periods_per_year):
        """
        Computes the annualized Sharpe ratio of a set of returns
        """
        excess_ret = FinancialMetrics._risk_free_adjusted_returns(r, riskfree_rate, periods_per_year)
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        ann_vol = FinancialMetrics.annualize_vol(r, periods_per_year)
        return ann_ex_ret / ann_vol if ann_vol != 0 else 0

    @staticmethod
    def rovar(r, periods_per_year, level=5):
        """
        Compute the Return on Value-at-Risk
        """
        return FinancialMetrics.annualize_rets(r, periods_per_year=periods_per_year) / abs(FinancialMetrics.var_historic(r, level=level)) if abs(FinancialMetrics.var_historic(r, level=level)) > 1e-10 else 0

    @staticmethod
    def sortino_ratio(r, riskfree_rate, periods_per_year):
        """
        Compute the annualized Sortino Ratio of a set of returns
        """
        excess_ret = FinancialMetrics._risk_free_adjusted_returns(r, riskfree_rate, periods_per_year)
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        neg_rets = excess_ret[excess_ret < 0]
        ann_vol = FinancialMetrics.annualize_vol(neg_rets, periods_per_year)
        return ann_ex_ret / ann_vol if ann_vol != 0 else 0

    @staticmethod
    def calmar_ratio(r, riskfree_rate, periods_per_year):
        """
        Compute the annualized Calmar Ratio of a set of returns
        """
        excess_ret = FinancialMetrics._risk_free_adjusted_returns(r, riskfree_rate, periods_per_year)
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        max_dd = abs(FinancialMetrics.drawdown(r).Drawdown.min())
        return ann_ex_ret / max_dd if max_dd != 0 else 0

    @staticmethod
    def burke_ratio(r, riskfree_rate, periods_per_year, modified=False):
        """
        Compute the annualized Burke Ratio of a set of returns
        If "modified" is True, then the modified Burke Ratio is returned
        """
        excess_ret = FinancialMetrics._risk_free_adjusted_returns(r, riskfree_rate, periods_per_year)
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        sum_dwn = np.sqrt(np.sum((FinancialMetrics.drawdown(r).Drawdown)**2))
        if not modified:
            bk_ratio = ann_ex_ret / sum_dwn if sum_dwn != 0 else 0
        else:
            bk_ratio = ann_ex_ret / sum_dwn * np.sqrt(len(r)) if sum_dwn != 0 else 0
        return bk_ratio

    @staticmethod
    def net_profit(returns):
        """
        Calculates the net profit of a strategy.
        """
        cumulative_returns = (1 + returns).cumprod() - 1
        return cumulative_returns.iloc[-1]

    @staticmethod
    def worst_drawdown(returns):
        """
        Calculates the worst drawdown from cumulative returns.
        """
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

    @staticmethod
    def summary_stats(r, riskfree_rate=0.03, periods_per_year=12):
        """
        Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
        """
        ann_r = FinancialMetrics._annualize(FinancialMetrics.annualize_rets, r, periods_per_year)
        ann_vol = FinancialMetrics._annualize(FinancialMetrics.annualize_vol, r, periods_per_year)
        semidev = FinancialMetrics._annualize(FinancialMetrics.semideviation, r, periods_per_year)
        ann_sr = FinancialMetrics._annualize(FinancialMetrics.sharpe_ratio, r, periods_per_year, riskfree_rate=riskfree_rate)
        ann_cr = FinancialMetrics._annualize(FinancialMetrics.calmar_ratio, r, periods_per_year, riskfree_rate=riskfree_rate)
        ann_br = FinancialMetrics._annualize(FinancialMetrics.burke_ratio, r, periods_per_year, riskfree_rate=riskfree_rate, modified=True)
        ann_sortr = FinancialMetrics._annualize(FinancialMetrics.sortino_ratio, r, periods_per_year, riskfree_rate=riskfree_rate)
        dd = r.aggregate(lambda r: FinancialMetrics.drawdown(r).Drawdown.min())
        skew = r.aggregate(FinancialMetrics.skewness)
        kurt = r.aggregate(FinancialMetrics.kurtosis)
        hist_var5 = r.aggregate(FinancialMetrics.var_historic)
        cf_var5 = r.aggregate(FinancialMetrics.var_gaussian, modified=True)
        hist_cvar5 = r.aggregate(FinancialMetrics.cvar_historic)
        rovar5 = r.aggregate(FinancialMetrics.rovar, periods_per_year=periods_per_year)
        np_wdd_ratio = r.aggregate(lambda returns: FinancialMetrics.net_profit(returns) / -FinancialMetrics.worst_drawdown(returns))
        return pd.DataFrame({
            "Annualized Return": round(ann_r, 4),
            "Annualized Volatility": round(ann_vol, 4),
            "Semi-Deviation": round(semidev, 4),
            "Skewness": round(skew, 4),
            "Kurtosis": round(kurt, 4),
            "Historic VaR (5%)": round(hist_var5, 4),
            "Cornish-Fisher VaR (5%)": round(cf_var5, 4),
            "Historic CVaR (5%)": round(hist_cvar5, 4),
            "Return on VaR": round(rovar5, 4),
            "Sharpe Ratio": round(ann_sr, 4),
            "Sortino Ratio": round(ann_sortr, 4),
            "Calmar Ratio": round(ann_cr, 4),
            "Modified Burke Ratio": round(ann_br, 4),
            "Max Drawdown": round(dd, 4),
            "Net Profit to Worst Drawdown": round(np_wdd_ratio, 4)
        })
    
    @staticmethod
    def portfolio_return(weights, returns):
        """
        Weights -> returns
        """
        return weights.T @ returns

    @staticmethod
    def portfolio_vol(weights, covmat):
        """
        Weights -> volatility
        """
        return (weights.T @ covmat @ weights)**0.5

    @staticmethod
    def tracking_error(r_a, r_b):
        """
        Returns the Tracking Error between the two return series
        """
        return np.sqrt(((r_a - r_b)**2).sum())

    @staticmethod
    def information_ratio(r_a, r_b):
        """
        Returns the Information Ratio between two return series.
        """
        diff = r_a - r_b
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        return mean_diff / std_diff


class PortfolioOptimization:
    @staticmethod
    def msr(riskfree_rate, er, cov):
        """
        Returns the weights of the portfolio that gives you the maximum Sharpe ratio
        given the riskfree rate, expected returns, and a covariance matrix
        """
        n = er.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0, 1.0),) * n
        weights_sum_to_1 = {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1
        }

        def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
            """
            -> Negative Sharpe ratio, given weights
            """
            r = FinancialMetrics.portfolio_return(weights, er)
            vol = FinancialMetrics.portfolio_vol(weights, cov)
            return -(r - riskfree_rate) / vol

        results = minimize(neg_sharpe_ratio, init_guess,
                           args=(riskfree_rate, er, cov,), method="SLSQP", 
                           options={"disp": False},
                           constraints=(weights_sum_to_1),
                           bounds=bounds)
        return results.x

    @staticmethod
    def gmv(cov):
        """
        Returns the weight of the Global Minimum Volatility portfolio, given a covariance matrix
        """
        n = cov.shape[0]
        return PortfolioOptimization.msr(0, np.repeat(1, n), cov)

    @staticmethod
    def sample_cov(r, **kwargs):
        """
        Returns the sample covariance of the supplied returns
        """
        return r.cov()

    @staticmethod
    def cc_cov(r, **kwargs):
        """
        Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
        """
        rhos = r.corr()
        n = rhos.shape[0]
        rho_bar = (rhos.values.sum() - n) / (n * (n - 1))
        ccor = np.full_like(rhos, rho_bar)
        np.fill_diagonal(ccor, 1.0)
        sd = r.std()
        ccov = ccor * np.outer(sd, sd)
        return pd.DataFrame(ccov, index=r.columns, columns=r.columns)

    @staticmethod
    def shrinkage_cov(r, delta=0.5, **kwargs):
        """
        Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators
        """
        prior = PortfolioOptimization.cc_cov(r, **kwargs)
        sample = PortfolioOptimization.sample_cov(r, **kwargs)
        return delta * prior + (1 - delta) * sample

    @staticmethod
    def risk_contribution(w, cov):
        """
        Compute the contributions to risk of the constituents of a portfolio, given a set of portfolio weights and a covariance matrix
        """
        total_portfolio_var = FinancialMetrics.portfolio_vol(w, cov)**2
        marginal_contrib = cov @ w
        risk_contrib = np.multiply(marginal_contrib, w.T) / total_portfolio_var
        return risk_contrib

    @staticmethod
    def equal_risk_contributions(cov):
        """
        Returns the weights of the portfolio that equalizes the contributions
        of the constituents based on the given covariance matrix
        """
        n = cov.shape[0]
        return PortfolioOptimization.target_risk_contributions(target_risk=np.repeat(1/n, n), cov=cov)

    @staticmethod
    def target_risk_contributions(target_risk, cov):
        """
        Equal Risk Contribution Portfolio Optimization
        """
        n = len(target_risk)
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0, 1.0),) * n
        weights_sum_to_1 = {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1
        }

        def risk_budget_objective(weights, target_risk, cov):
            risk_contributions = PortfolioOptimization.risk_contribution(weights, cov)
            return np.sum((risk_contributions - target_risk)**2)
        
        results = minimize(risk_budget_objective, init_guess,
                           args=(target_risk, cov,), method="SLSQP", 
                           options={"disp": False},
                           constraints=(weights_sum_to_1),
                           bounds=bounds)
        return results.x

    @staticmethod
    def lz_complexity(s):
        """Calculate the Lempel-Ziv complexity of a binary sequence"""
        import zlib
        s = s.encode('utf-8')
        compressed_length = len(zlib.compress(s))
        return compressed_length

    @staticmethod
    def preprocess_series(series):
        """Convert series to a binary sequence"""
        median_val = series.median()
        return ''.join('1' if x > median_val else '0' for x in series)


class PortfolioWeighting:
    @staticmethod
    def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
        """
        Returns the weights of the EW portfolio based on the asset returns "r" as a DataFrame
        If supplied a set of capweights and a capweight tether, it is applied and reweighted 
        """
        n = len(r.columns)
        ew = pd.Series(1/n, index=r.columns)
        if cap_weights is not None:
            cw = cap_weights.loc[r.index[0]]
            if microcap_threshold is not None and microcap_threshold > 0:
                microcap = cw < microcap_threshold
                ew[microcap] = 0
                ew = ew / ew.sum()
            if max_cw_mult is not None and max_cw_mult > 0:
                ew = np.minimum(ew, cw * max_cw_mult)
                ew = ew / ew.sum()
        return ew

    @staticmethod
    def weight_cw(r, cap_weights):
        """
        Returns the weights of the CW portfolio based on the time series of cap weights
        """
        last_mcap = cap_weights.loc[r.index[-1]]
        weights = last_mcap / last_mcap.sum()
        return weights

    @staticmethod
    def weight_gmv(r, cov_estimator=None, **kwargs):
        """
        Produces the weights of the GMV portfolio given a covariance matrix of the returns 
        """
        est_cov = cov_estimator(r, **kwargs) if cov_estimator else r.cov()
        return PortfolioOptimization.gmv(est_cov)

    @staticmethod
    def weight_erc(r, cov_estimator=None, **kwargs):
        """
        Produces the weights of the ERC portfolio given a covariance matrix of the returns 
        """
        est_cov = cov_estimator(r, **kwargs) if cov_estimator else r.cov()
        return PortfolioOptimization.equal_risk_contributions(est_cov)

    @staticmethod
    def weight_min_corr(r):
        """
        Returns the weights of the portfolio that minimizes the sum of all pairwise correlations
        """
        corr_matrix = r.corr()
        n = len(corr_matrix)
        init_guess = np.random.uniform(0, 1.0, n)
        init_guess /= init_guess.sum()
        bounds = tuple((0, 1.0) for _ in range(n))

        def portfolio_corr(weights):
            weighted_corr_matrix = np.outer(weights, weights) * corr_matrix
            total_corr = np.sum(np.triu(weighted_corr_matrix, 1))
            return total_corr

        constraints = ({
            'type': 'eq', 
            'fun': lambda weights: np.sum(weights) - 1
        })
        options = {'ftol': 1e-6, 'disp': False}
        result = minimize(portfolio_corr, init_guess, method='SLSQP', bounds=bounds, constraints=constraints, options=options)
        if result.success:
            return pd.Series(result.x, index=r.columns)
        else:
            raise Exception('Optimization did not converge')

    @staticmethod
    def weight_max_div(r):
        """
        Returns the weights of the portfolio that maximizes the diversification ratio
        The diversification ratio is defined as the weighted average volatility divided by the portfolio volatility
        """
        cov_matrix = r.cov()
        std_devs = np.sqrt(np.diag(cov_matrix))
        
        def diversification_ratio(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            weighted_vol = np.sum(weights * std_devs)
            div_ratio = weighted_vol / portfolio_vol
            return -div_ratio
        
        n = len(std_devs)
        init_guess = np.repeat(1/n, n)
        bounds = tuple((0, 1) for _ in range(n))
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        result = minimize(diversification_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': False})
        if result.success:
            return pd.Series(result.x, index=r.columns)
        else:
            raise BaseException('Optimization did not converge: ' + result.message)

    @staticmethod
    def weight_mom(r, lookback_period):
        """
        Calculate portfolio weights based on cross-sectional momentum.
        lookback_period: Integer, the number of periods to look back to calculate momentum
        """
        momentum = r.rolling(window=lookback_period).apply(lambda x: np.prod(1 + x) - 1, raw=False)
        momentum = momentum.dropna()
        if momentum.empty:
            return pd.Series(1.0 / len(r.columns), index=r.columns)
        rankings = momentum.rank(axis=1, ascending=False)
        weights = 1 / rankings
        normalized_weights = weights.div(weights.sum(axis=1), axis=0)
        if normalized_weights.empty:
            return pd.Series(1.0 / len(r.columns), index=r.columns)
        else:
            return normalized_weights.iloc[-1]

    @staticmethod
    def weight_kmeans(r, n_clusters):
        """
        Generate portfolio weights using K-means clustering on asset returns
        """
        r_transposed = r.transpose().dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(r_transposed)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(scaled_data)
        labels = kmeans.labels_
        weights = pd.Series(index=r.columns, dtype=float)
        for i in range(n_clusters):
            cluster_assets = r_transposed.index[labels == i]
            weights[cluster_assets] = 1 / len(cluster_assets)
        return weights / weights.sum()

    @staticmethod
    def weight_hier(r, n_clusters):
        """
        Generate portfolio weights using hierarchical clustering on asset returns
        """
        r_transposed = r.transpose().dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(r_transposed)
        Z = linkage(scaled_data, method='ward')
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        weights = pd.Series(0, index=r.columns, dtype='float64')
        for i in range(1, n_clusters + 1):
            cluster_assets = r_transposed.index[labels == i]
            weights[cluster_assets] = 1 / len(cluster_assets)
        return weights / weights.sum()
    
    @staticmethod
    def weight_hrp(r):
        """
        Generate portfolio weights using Hierarchical Risk Parity (HRP) with automatic determination of clusters
        """
        cov_matrix = r.cov()
        Z = linkage(cov_matrix, method='ward')

        def find_optimal_num_clusters(Z):
            """
            Find the optimal number of clusters using the dendrogram from hierarchical clustering
            """
            last_merges = Z[-10:, 2]
            jump = np.diff(last_merges)
            jump_rev = jump[::-1]
            d_rev = jump_rev.argmax() + 2
            return len(last_merges) - d_rev + 2
        
        optimal_num_clusters = find_optimal_num_clusters(Z)
        labels = fcluster(Z, optimal_num_clusters, criterion='maxclust')
        weights = pd.Series(0, index=r.columns, dtype='float64')
        for i in range(1, optimal_num_clusters + 1):
            cluster_assets = r.columns[labels == i]
            sub_cov_matrix = cov_matrix.loc[cluster_assets, cluster_assets]
            inv_diag = 1 / np.diag(sub_cov_matrix.values)
            cluster_weights = inv_diag / np.sum(inv_diag)
            weights[cluster_assets] = cluster_weights
        weights /= weights.sum()
        return weights

    @staticmethod
    def weight_pca(r, n_components):
        """
        Generate portfolio weights using PCA on asset returns
        """
        scaler = StandardScaler()
        r_scaled = scaler.fit_transform(r)
        pca = PCA(n_components=n_components)
        pca.fit(r_scaled)
        loadings = pca.components_[0]
        abs_loadings = np.abs(loadings)
        weights = abs_loadings / np.sum(abs_loadings)
        return pd.Series(weights, index=r.columns)

    @staticmethod
    def weight_trp(returns, alpha):
        """
        Generate portfolio weights based on Tail Risk Parity approach
        """
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns.fillna(0))
        scaled_returns = pd.DataFrame(scaled_returns, columns=returns.columns)
        cvars = scaled_returns.apply(lambda x: FinancialMetrics.cvar_historic(x, level=alpha))

        def objective(weights):
            portfolio_returns = (scaled_returns * weights).sum(axis=1)
            portfolio_cvar = FinancialMetrics.cvar_historic(portfolio_returns, level=alpha)
            weighted_cvars = (weights * cvars).sum()
            return abs(portfolio_cvar - weighted_cvars)
        
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0, 1) for _ in returns.columns)
        init_weights = np.array([1 / len(returns.columns)] * len(returns.columns))
        result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        if not result.success:
            raise BaseException("Optimization failed")
        optimal_weights = pd.Series(result.x, index=returns.columns)
        return optimal_weights

    @staticmethod
    def weight_wavelet(returns, wavelet_name, max_level):
        """
        Calculate portfolio weights based on the wavelet variance of each asset's returns
        """
        wavelet = pywt.Wavelet(wavelet_name)
        
        def compute_wavelet_variance(ts):
            ts_clean = ts.dropna()
            if len(ts_clean) < 2:
                return np.nan
            max_dec_level = pywt.dwt_max_level(len(ts_clean), wavelet.dec_len)
            max_dec_level = min(max_level, max_dec_level)
            coeffs = pywt.wavedec(ts_clean, wavelet, level=max_dec_level)
            variance = [np.var(arr) for arr in coeffs if len(arr) > 1]
            return sum(variance)

        wavelet_variances = returns.apply(compute_wavelet_variance, axis=0)
        wavelet_variances = wavelet_variances.replace(0, np.nan).fillna(np.nanmean(wavelet_variances))
        inverse_variances = 1 / wavelet_variances
        weights = inverse_variances / inverse_variances.sum()
        return weights
    
    @staticmethod
    def weight_msr(returns, risk_free_rate, periods_per_year):
        """
        Function to optimize the portfolio by maximizing the Sharpe ratio using historical return data
        """
        
        def annualized_performance(weights, mean_returns, cov_matrix):
            """
            Calculate the portfolio performance metrics
            """
            returns = np.sum(mean_returns * weights) * periods_per_year
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(periods_per_year)
            return returns, volatility

        def negative_sharpe_ratio(weights, mean_returns, cov_matrix):
            """
            Return the negative of Sharpe ratio, since we are minimizing
            """
            returns, volatility = annualized_performance(weights, mean_returns, cov_matrix)
            return -(returns - risk_free_rate) / volatility

        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = (0.0, 1.0)
        bounds = tuple(bound for asset in range(num_assets))
        initial_guess = num_assets * [1. / num_assets]
        result = minimize(negative_sharpe_ratio, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            optimized_weights = pd.Series(result.x, index=returns.columns)
            return optimized_weights
        else:
            raise BaseException('Optimization did not converge')

    @staticmethod
    def weight_algo(returns):
        """
        Calculate portfolio weights based on the Lempel-Ziv complexity of each asset's returns
        """
        complexities = {}
        for column in returns.columns:
            series = PortfolioOptimization.preprocess_series(returns[column])
            complexities[column] = PortfolioOptimization.lz_complexity(series)
        complexity_series = pd.Series(complexities)
        inverse_complexities = 1 / complexity_series
        weights = inverse_complexities / inverse_complexities.sum()
        return weights

    @staticmethod
    def weight_rf(returns, lookback_periods):
        """
        Generate portfolio weights using Random Forest based on historical return data
        """
        shifted_data = []
        for i in range(1, lookback_periods + 1):
            shifted = returns.shift(i)
            shifted.columns = [f'{col}_lag_{i}' for col in returns.columns]
            shifted_data.append(shifted)
        features = pd.concat(shifted_data, axis=1)
        features.dropna(inplace=True)
        targets = returns.shift(-1).dropna()
        valid_indices = features.index.intersection(targets.index)
        features = features.loc[valid_indices]
        targets = targets.loc[valid_indices]
        rf = RandomForestRegressor(n_estimators=100, random_state=42, verbose=0)
        rf.fit(features, targets)
        importances = rf.feature_importances_
        asset_importances = {}
        for i, col in enumerate(returns.columns):
            asset_importances[col] = np.mean(importances[i::len(returns.columns)])
        weights = pd.Series(asset_importances).div(sum(asset_importances.values()))
        return weights


class PortfolioAnalysis:
    @staticmethod
    def plot_cumulative_returns(data, title="Portfolio Evolution"):
        """
        Plots the cumulative returns of a portfolio interactively using Plotly
        """
        fig = go.Figure()
        for col in data.columns:
            cumulative_returns = (1 + data[col]).cumprod()
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns,
                mode='lines',
                name=col
            ))
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Cumulative Returns',
            hovermode='x unified'
        )
        fig.show()

    @staticmethod
    def plot_weights(weights, title="Portfolio Weights", figsize=(15, 6)):
        """
        Plots the weights of the portfolio
        """
        weights.T.plot.bar(stacked=True, figsize=figsize, legend=False)
        plt.title(title)
        plt.show()

    @staticmethod
    def print_tracking_errors(cwr, **kwargs):
        """
        Prints the tracking error between a source return series and multiple target return series
        """
        for strategy, returns in kwargs.items():
            te = FinancialMetrics.tracking_error(returns, cwr)
            print(f"Tracking error for {strategy} is: {te:.4f}")

    @staticmethod
    def print_information_ratios(cwr, **kwargs):
        """
        Prints the information ratio between a source return series and multiple target return series
        """
        for strategy, returns in kwargs.items():
            ir = FinancialMetrics.information_ratio(returns, cwr)
            print(f"Information Ratio for {strategy} is: {ir:.4f}")

