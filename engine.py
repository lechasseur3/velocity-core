import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import BlackLittermanModel, risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
import scipy.stats as stats
import concurrent.futures

def fetch_stock_data(ticker):
    """Fetch market cap, debt, and rich fundamentals for a given ticker."""
    stock = yf.Ticker(ticker)
    info_dict = {
        'ticker': ticker,
        'mcap': 1e9,
        'total_debt': 0,
        'currency': 'USD',
        'price': 0,
        'change_percent': 0,
        'pe_ratio': None,
        'dividend_yield': 0,
        'quoteType': 'EQUITY'
    }
    
    try:
        info = stock.info
        info_dict['mcap'] = info.get('marketCap', 1e9)
        info_dict['currency'] = info.get('currency', 'USD')
        
        # Fundamentals
        info_dict['price'] = info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose', 0)))
        info_dict['pe_ratio'] = info.get('trailingPE', info.get('forwardPE', None))
        
        # Yield (can come as 0.05 for 5% or None)
        dy = info.get('dividendYield', 0)
        info_dict['dividend_yield'] = float(dy) if dy is not None else 0.0
        
        info_dict['quoteType'] = info.get('quoteType', 'EQUITY')
        
        try:
            dp = info.get('regularMarketChangePercent', 0)
            info_dict['change_percent'] = float(dp) if dp is not None else 0.0
        except: pass
        
        try:
            info_dict['total_debt'] = stock.balance_sheet.loc['Total Debt'].iloc[0]
        except: pass
        
    except Exception:
        pass
        
    return info_dict

def get_fama_french_exposure(portfolio_returns, start_date, end_date):
    """Run an OLS regression on the portfolio returns against Kenneth French's 5 Global Factors."""
    try:
        import pandas_datareader.data as web
        import statsmodels.api as sm
    except ImportError:
        return None
        
    try:
        ff5 = web.DataReader('Global_5_Factors_Daily', 'famafrench', start_date, end_date)[0] / 100
        if ff5.index.tz is not None:
            ff5.index = ff5.index.tz_localize(None)
            
        # Ensure our returns are tz-naive for alignment
        if portfolio_returns.index.tz is not None:
            portfolio_returns.index = portfolio_returns.index.tz_localize(None)
            
        common = portfolio_returns.index.intersection(ff5.index)
        if len(common) < 10:
            return None
            
        y = portfolio_returns.loc[common] - ff5.loc[common, 'RF']
        X = sm.add_constant(ff5.loc[common, ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])
        model = sm.OLS(y, X, missing='drop').fit()
        
        return {
            'Mkt-RF': float(model.params['Mkt-RF']),
            'SMB': float(model.params['SMB']),
            'HML': float(model.params['HML']),
            'RMW': float(model.params['RMW']),
            'CMA': float(model.params['CMA'])
        }
    except Exception as e:
        print(f"Fama-French Error: {e}")
        return None

def get_portfolio_data(symbols):
    """Fetch data for all symbols."""
    # Historical Prices
    tickers = yf.Tickers(symbols)
    df = tickers.history(period='5y', auto_adjust=True)['Close']
    
    # Check if empty
    if df.empty or len(df) < 5:
        # Emergency fallback
        raise ValueError("Data fetching failed: No price history returned. Please try again in a few seconds.")
    
    df = df.dropna()
    
    # Fundamentals (Parallel)
    mcaps = {}
    debts = {}
    currencies = {}
    assets_info = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(fetch_stock_data, t): t for t in symbols}
        for future in concurrent.futures.as_completed(future_to_ticker):
            res = future.result()
            t = res['ticker']
            mcaps[t] = res['mcap']
            debts[t] = res['total_debt']
            currencies[t] = res['currency']
            assets_info[t] = res
            
    fetched_symbols = df.columns.tolist()
    unique_currencies = set([currencies.get(s, 'USD') for s in fetched_symbols]) - {'USD'}
    if unique_currencies:
        fx_pairs = [f"{cur}USD=X" for cur in unique_currencies]
        fx_data = yf.download(fx_pairs, period='5y', auto_adjust=True, progress=False)['Close']
        if isinstance(fx_data, pd.Series):
            fx_data = fx_data.to_frame(fx_pairs[0])
            
        for ticker in fetched_symbols:
            cur = currencies.get(ticker, 'USD')
            if cur != 'USD':
                fx_pair = f"{cur}USD=X"
                if fx_pair in fx_data.columns:
                    fx = fx_data[fx_pair].ffill().bfill()
                    aligned_fx = fx.reindex(df.index).ffill().bfill()
                    df[ticker] = df[ticker] * aligned_fx
                
    return df, mcaps, debts, assets_info

def run_analysis(symbols, views, is_auto=True, manual_weights=None):
    """
    symbols: list of strings
    views: list of dicts {type: 'A'|'R', asset: idx, bull: idx, bear: idx, value: float}
    is_auto: bool
    manual_weights: dict {symbol: weight_percent}
    """
    df, mcaps, debts, assets_info = get_portfolio_data(symbols)
    df100 = df / df.iloc[0] * 100
    
    # Risk-free rate
    try:
        rf_data = yf.Ticker('^IRX').history(period='5d')['Close']
        rf = rf_data.dropna().mean() / 100 if not rf_data.empty else 0.04
        if np.isnan(rf): rf = 0.04
    except:
        rf = 0.04

    # DATA ALIGNMENT: Ensure symbols match the fetched data
    fetched_symbols = df.columns.tolist()
    n_assets = len(fetched_symbols)
    
    # Filter mcaps to match fetched data
    aligned_mcaps = {s: mcaps.get(s, 1e9) for s in fetched_symbols}
    S = risk_models.sample_cov(df)
    
    # Black-Litterman
    bl_returns = None
    if len(views) > 0:
        P_list = []
        Q_list = []
        for v in views:
            row = np.zeros(n_assets)
            try:
                if v.get('type') == 'A' and v.get('asset') is not None:
                    ticker = symbols[v['asset']]
                    if ticker in fetched_symbols:
                        row[fetched_symbols.index(ticker)] = 1.0
                        P_list.append(row)
                        Q_list.append(v['value'] / 100.0)
                elif v.get('type') == 'R' and v.get('bull') is not None and v.get('bear') is not None:
                    t_bull = symbols[v['bull']]
                    t_bear = symbols[v['bear']]
                    if t_bull in fetched_symbols and t_bear in fetched_symbols:
                        row[fetched_symbols.index(t_bull)] = 1.0
                        row[fetched_symbols.index(t_bear)] = -1.0
                        P_list.append(row)
                        Q_list.append(v['value'] / 100.0)
            except (IndexError, KeyError):
                continue
        
        if len(P_list) > 0:
            P = np.array(P_list)
            Q = np.array(Q_list).reshape(-1, 1)
            bl = BlackLittermanModel(S, pi="market", market_caps=aligned_mcaps, risk_aversion=2.5, P=P, Q=Q, tau=0.05)
            bl_returns = bl.bl_returns()
    
    if bl_returns is None:
        from pypfopt import black_litterman
        bl_returns = black_litterman.market_implied_prior_returns(cov_matrix=S, risk_aversion=2.5, market_caps=aligned_mcaps)
    
    # Final check for Series type and alignment
    if not isinstance(bl_returns, pd.Series):
        bl_returns = pd.Series(bl_returns, index=fetched_symbols)
    bl_returns = bl_returns.reindex(fetched_symbols).fillna(0)
    
    # Optimization
    if is_auto:
        ef = EfficientFrontier(bl_returns, S)
        ef.max_sharpe(risk_free_rate=rf)
        weights = ef.clean_weights()
    else:
        # Manual weights handling
        weights = {s: 0.125 for s in fetched_symbols} # default
        if manual_weights:
            total = sum(float(v) for v in manual_weights.values()) or 1.0
            weights = {s: float(manual_weights.get(s, 0)) / total for s in fetched_symbols}
        ef = EfficientFrontier(bl_returns, S) # Still create EF for performance metrics
    
    perf = ef.portfolio_performance(risk_free_rate=rf) if is_auto else None
    
    # If manual, we need to calculate performance for these specific weights
    if not is_auto:
        w_series = pd.Series(weights).reindex(fetched_symbols).fillna(0)
        expected_return = np.dot(w_series, bl_returns)
        volatility = np.sqrt(np.dot(w_series.T, np.dot(S, w_series)))
        sharpe = (expected_return - rf) / volatility if volatility > 0 else 0
        perf = (expected_return, volatility, sharpe)
    
    # Risk Analysis
    w_series = pd.Series(weights).reindex(fetched_symbols).fillna(0)
    returns_df = df.pct_change().dropna()
    rp = (returns_df[fetched_symbols] * w_series).sum(axis=1)
    
    # Benchmark Comparison & Evolution
    try:
        benchmark_ticker = yf.Ticker('SPY')
        spy_hist = benchmark_ticker.history(period='5y')['Close'].dropna()
        
        spy_returns = spy_hist.pct_change().dropna()
        if spy_returns.index.tz is not None:
            spy_returns.index = spy_returns.index.tz_localize(None)
        if rp.index.tz is not None:
            rp.index = rp.index.tz_localize(None)
            
        aligned = pd.concat([rp, spy_returns], axis=1).dropna()
        if len(aligned) > 5:
            cov_mat = np.cov(aligned.iloc[:,0], aligned.iloc[:,1])
            var_bench = np.var(aligned.iloc[:,1])
            beta = cov_mat[0, 1] / var_bench if var_bench > 1e-9 else 1.0
        else:
            beta = 1.0
            
        # Benchmark Evolution (Normalized to 100 for Charting)
        if spy_hist.index.tz is not None:
            spy_hist.index = spy_hist.index.tz_localize(None)
        
        # Ensure df100 is tz-naive
        if df100.index.tz is not None:
            df100.index = df100.index.tz_localize(None)
            
        common_idx = df100.index.intersection(spy_hist.index)
        if len(common_idx) > 0:
            spy_aligned = spy_hist.loc[common_idx]
            spy_aligned = spy_aligned / spy_aligned.iloc[0] * 100
            
            # Format to dicts
            benchmark_evolution = pd.DataFrame({'Date': common_idx, 'SPY': spy_aligned.values}).to_dict(orient='records')
        else:
            benchmark_evolution = []
            
    except:
        beta = 1.0
        benchmark_evolution = []
    
    alpha = perf[0] - (rf + beta * (0.12 - rf)) # Fallback to 12% market return if benchmark fails
    
    # VaR Calculation (1-Year 99% VaR, Normalized to 1.0)
    # Using annual parameters: expected return and volatility
    # Z-score for 99% is approx 2.326
    var_99_pct = -(perf[0] - 2.326 * perf[1])
    var_99_pct = max(0, var_99_pct) # Ensure non-negative risk
    
    # Efficient Frontier
    vols, rets = [], []
    try:
        r_min, r_max = bl_returns.min(), bl_returns.max()
        if r_min < r_max:
            for t in np.linspace(r_min, r_max, 20):
                try:
                    ef_t = EfficientFrontier(bl_returns, S)
                    ef_t.efficient_return(t)
                    r, v, _ = ef_t.portfolio_performance()
                    rets.append(float(r)); vols.append(float(v))
                except: pass
    except: pass

    corr_matrix = returns_df[fetched_symbols].corr().round(3).values.tolist()
    
    # Fama-French Exposure
    ff_exposure = get_fama_french_exposure(rp, df.index[0], df.index[-1])

    return {
        "weights": weights,
        "performance": {
            "expected_return": float(perf[0]),
            "volatility": float(perf[1]),
            "sharpe": float(perf[2]),
            "beta": float(beta),
            "alpha": float(alpha) if not np.isnan(alpha) else 0.0,
            "var_99_pct": float(var_99_pct)
        },
        "risk_contribution": (w_series * (S @ w_series) / (perf[1]**2 if perf[1] > 1e-6 else 1.0)).to_dict(),
        "historical_evolution": df100.reset_index().to_dict(orient='records'),
        "benchmark_evolution": benchmark_evolution,
        "daily_returns": returns_df.reset_index().to_dict(orient='records'),
        "efficient_frontier": {"vols": vols, "rets": rets},
        "correlation_matrix": corr_matrix,
        "assets": fetched_symbols,
        "fama_french": ff_exposure,
        "assets_info": assets_info
    }
