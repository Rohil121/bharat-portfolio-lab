
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Bharat Portfolio Lab v0.3.1",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------

st.title("📊 Bharat Portfolio Lab")

st.subheader("India 10 Adaptive Barbell Portfolio")

st.markdown(
    """
    **Version v0.3.1 — Complete Robustness and Attribution**

    This version analyses a 10-stock Indian equity portfolio and compares:

    - the Adaptive Barbell strategy,
    - the fixed equal-weighted India 10 portfolio, and
    - the Nifty 50 benchmark.
    """
)

st.info(
    "This application is an academic financial-analytics project "
    "and does not constitute investment advice."
)



# ---------------------------------------------------------
# Precomputed robustness data
# ---------------------------------------------------------

APP_DIRECTORY = Path(__file__).resolve().parent

ROBUSTNESS_DATA_DIRECTORY = (
    APP_DIRECTORY
    / "data"
    / "processed"
    / "robustness"
)


@st.cache_data
def load_robustness_data():
    """
    Load the precomputed v0.3 robustness-test outputs.
    """

    file_paths = {
        "development_performance":
            ROBUSTNESS_DATA_DIRECTORY
            / "development_performance.csv",

        "out_of_sample_performance":
            ROBUSTNESS_DATA_DIRECTORY
            / "out_of_sample_performance.csv",

        "parameter_sensitivity":
            ROBUSTNESS_DATA_DIRECTORY
            / "parameter_sensitivity.csv",

        "transaction_cost_stress":
            ROBUSTNESS_DATA_DIRECTORY
            / "transaction_cost_stress.csv",

        "rolling_sharpe":
            ROBUSTNESS_DATA_DIRECTORY
            / "rolling_sharpe.csv",

        "rolling_volatility":
            ROBUSTNESS_DATA_DIRECTORY
            / "rolling_volatility.csv",

        "rolling_drawdown":
            ROBUSTNESS_DATA_DIRECTORY
            / "rolling_drawdown.csv",

        "rolling_beta":
            ROBUSTNESS_DATA_DIRECTORY
            / "rolling_beta.csv",

        "market_regime_performance":
            ROBUSTNESS_DATA_DIRECTORY
            / "market_regime_performance.csv",

        "robustness_scorecard":
            ROBUSTNESS_DATA_DIRECTORY
            / "robustness_scorecard.csv",

        "research_limitations":
            ROBUSTNESS_DATA_DIRECTORY
            / "research_limitations.csv",

        "final_robustness_assessment":
            ROBUSTNESS_DATA_DIRECTORY
            / "final_robustness_assessment.csv",

        "attribution_reconstruction_diagnostic":
            ROBUSTNESS_DATA_DIRECTORY
            / "attribution_reconstruction_diagnostic.csv",

        "bucket_attribution":
            ROBUSTNESS_DATA_DIRECTORY
            / "bucket_attribution.csv",

        "bull_bear_performance":
            ROBUSTNESS_DATA_DIRECTORY
            / "bull_bear_performance.csv",

        "classification_evidence":
            ROBUSTNESS_DATA_DIRECTORY
            / "classification_evidence.csv",

        "counterfactual_performance":
            ROBUSTNESS_DATA_DIRECTORY
            / "counterfactual_performance.csv",

        "final_robustness_classification":
            ROBUSTNESS_DATA_DIRECTORY
            / "final_robustness_classification.csv",

        "holding_attribution":
            ROBUSTNESS_DATA_DIRECTORY
            / "holding_attribution.csv",

        "incremental_attribution":
            ROBUSTNESS_DATA_DIRECTORY
            / "incremental_attribution.csv",

        "sector_attribution":
            ROBUSTNESS_DATA_DIRECTORY
            / "sector_attribution.csv",

        "volatility_state_performance":
            ROBUSTNESS_DATA_DIRECTORY
            / "volatility_state_performance.csv",
    }

    missing_files = [
        file_path.name
        for file_path in file_paths.values()
        if not file_path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing robustness files: "
            + ", ".join(missing_files)
        )

    return {
        "development_performance": pd.read_csv(
            file_paths["development_performance"],
            index_col="Metric",
        ),

        "out_of_sample_performance": pd.read_csv(
            file_paths["out_of_sample_performance"],
            index_col="Metric",
        ),

        "parameter_sensitivity": pd.read_csv(
            file_paths["parameter_sensitivity"],
        ),

        "transaction_cost_stress": pd.read_csv(
            file_paths["transaction_cost_stress"],
        ),

        "rolling_sharpe": pd.read_csv(
            file_paths["rolling_sharpe"],
            parse_dates=["Date"],
            index_col="Date",
        ),

        "rolling_volatility": pd.read_csv(
            file_paths["rolling_volatility"],
            parse_dates=["Date"],
            index_col="Date",
        ),

        "rolling_drawdown": pd.read_csv(
            file_paths["rolling_drawdown"],
            parse_dates=["Date"],
            index_col="Date",
        ),

        "rolling_beta": pd.read_csv(
            file_paths["rolling_beta"],
            parse_dates=["Date"],
            index_col="Date",
        ),

        "market_regime_performance": pd.read_csv(
            file_paths["market_regime_performance"],
        ),

        "robustness_scorecard": pd.read_csv(
            file_paths["robustness_scorecard"],
        ),

        "research_limitations": pd.read_csv(
            file_paths["research_limitations"],
        ),

        "final_robustness_assessment": pd.read_csv(
            file_paths["final_robustness_assessment"],
            index_col="Metric",
        ),

        "attribution_reconstruction_diagnostic": pd.read_csv(
            file_paths["attribution_reconstruction_diagnostic"],
            index_col="Weight Convention",
        ),

        "bucket_attribution": pd.read_csv(
            file_paths["bucket_attribution"],
            index_col="Portfolio Bucket",
        ),

        "bull_bear_performance": pd.read_csv(
            file_paths["bull_bear_performance"],
        ),

        "classification_evidence": pd.read_csv(
            file_paths["classification_evidence"],
        ),

        "counterfactual_performance": pd.read_csv(
            file_paths["counterfactual_performance"],
            index_col="Metric",
        ),

        "final_robustness_classification": pd.read_csv(
            file_paths["final_robustness_classification"],
            index_col="Metric",
        ),

        "holding_attribution": pd.read_csv(
            file_paths["holding_attribution"],
            index_col="Ticker",
        ),

        "incremental_attribution": pd.read_csv(
            file_paths["incremental_attribution"],
            index_col="Effect",
        ),

        "sector_attribution": pd.read_csv(
            file_paths["sector_attribution"],
            index_col="Sector",
        ),

        "volatility_state_performance": pd.read_csv(
            file_paths["volatility_state_performance"],
        ),
    }


try:

    robustness_data = load_robustness_data()

except Exception as error:

    st.error(
        f"Robustness-test data could not be loaded: {error}"
    )

    st.stop()


# ---------------------------------------------------------
# Flagship portfolio
# ---------------------------------------------------------

portfolio = pd.DataFrame(
    {
        "Company": [
            "HDFC Bank",
            "Tata Consultancy Services",
            "Hindustan Unilever",
            "Sun Pharmaceutical Industries",
            "Power Grid Corporation of India",
            "Bharti Airtel",
            "Larsen & Toubro",
            "Mahindra & Mahindra",
            "Bharat Electronics",
            "Trent",
        ],
        "Ticker": [
            "HDFCBANK.NS",
            "TCS.NS",
            "HINDUNILVR.NS",
            "SUNPHARMA.NS",
            "POWERGRID.NS",
            "BHARTIARTL.NS",
            "LT.NS",
            "M&M.NS",
            "BEL.NS",
            "TRENT.NS",
        ],
        "Sector": [
            "Financial Services",
            "Information Technology",
            "Consumer Staples",
            "Healthcare",
            "Utilities",
            "Telecommunication",
            "Industrials",
            "Automobile",
            "Defence Electronics",
            "Consumer Retail",
        ],
        "Strategy Bucket": [
            "Resilient Compounder",
            "Resilient Compounder",
            "Resilient Compounder",
            "Resilient Compounder",
            "Resilient Compounder",
            "Growth Leader",
            "Growth Leader",
            "Growth Leader",
            "Growth Leader",
            "Growth Leader",
        ],
    }
)

portfolio["Initial Weight"] = 1 / len(portfolio)

portfolio_tickers = portfolio["Ticker"].tolist()

resilient_tickers = portfolio.loc[
    portfolio["Strategy Bucket"] == "Resilient Compounder",
    "Ticker",
].tolist()

growth_tickers = portfolio.loc[
    portfolio["Strategy Bucket"] == "Growth Leader",
    "Ticker",
].tolist()


# ---------------------------------------------------------
# Model constants
# ---------------------------------------------------------

BENCHMARK_TICKER = "^NSEI"
BENCHMARK_NAME = "Nifty 50"

TRADING_DAYS = 252
RISK_FREE_RATE = 0.065

MOMENTUM_LOOKBACK = 126
VOLATILITY_LOOKBACK = 63
TREND_LOOKBACK = 200

FULL_EQUITY_EXPOSURE = 1.00
DEFENSIVE_EQUITY_EXPOSURE = 0.70

TRANSACTION_COST_RATE = 0.0015


# ---------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------

st.sidebar.header("Model Settings")

start_date = st.sidebar.date_input(
    "Historical data start date",
    value=pd.Timestamp("2021-04-01"),
)

initial_capital = st.sidebar.number_input(
    "Initial investment (₹)",
    min_value=100_000,
    max_value=100_000_000,
    value=1_000_000,
    step=100_000,
)

st.sidebar.markdown(
    """
    ### Strategy settings

    - Monthly rebalancing
    - 6-month momentum
    - 3-month volatility
    - Nifty 50 200-day trend filter
    - 0.15% transaction-cost assumption
    """
)


# ---------------------------------------------------------
# Historical market data
# ---------------------------------------------------------

@st.cache_data(ttl=3600)
def download_market_data(
    tickers,
    benchmark_ticker,
    selected_start_date,
):
    all_tickers = tickers + [benchmark_ticker]

    download_end_date = (
        pd.Timestamp.today().normalize()
        + pd.Timedelta(days=1)
    )

    raw_data = yf.download(
        tickers=all_tickers,
        start=str(selected_start_date),
        end=download_end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw_data.empty:
        raise ValueError(
            "No market data was returned for the selected period."
        )

    if isinstance(raw_data.columns, pd.MultiIndex):

        if "Close" not in raw_data.columns.get_level_values(0):
            raise ValueError(
                "Adjusted closing prices were not returned."
            )

        prices = raw_data["Close"].copy()

    else:
        prices = raw_data[["Close"]].copy()
        prices.columns = all_tickers

    prices = (
        prices
        .reindex(columns=all_tickers)
        .sort_index()
    )

    completely_missing_tickers = (
        prices.columns[
            prices.isna().all()
        ].tolist()
    )

    if completely_missing_tickers:
        raise ValueError(
            "No price history was found for: "
            + ", ".join(completely_missing_tickers)
        )

    prices = (
        prices
        .ffill(limit=3)
        .dropna()
    )

    if len(prices) <= TREND_LOOKBACK:
        raise ValueError(
            "The selected period does not contain enough data "
            "for the 200-day trend calculation. "
            "Choose an earlier start date."
        )

    return prices


try:

    with st.spinner(
        "Downloading and validating Indian market data..."
    ):

        close_prices = download_market_data(
            portfolio_tickers,
            BENCHMARK_TICKER,
            start_date,
        )

except Exception as error:

    st.error(
        f"Market data could not be prepared: {error}"
    )

    st.stop()


stock_prices = (
    close_prices[portfolio_tickers]
    .copy()
)

benchmark_prices = (
    close_prices[[BENCHMARK_TICKER]]
    .rename(
        columns={
            BENCHMARK_TICKER: BENCHMARK_NAME
        }
    )
)

if stock_prices.shape[1] != 10:

    st.error(
        "Historical data is not available for all 10 holdings."
    )

    st.stop()


st.caption(
    f"Market data period: "
    f"{close_prices.index.min().date()} to "
    f"{close_prices.index.max().date()} | "
    f"{len(close_prices):,} trading observations"
)


# ---------------------------------------------------------
# Adaptive Barbell strategy signals
# ---------------------------------------------------------

stock_daily_returns = stock_prices.pct_change()

momentum_6m = (
    stock_prices
    .div(stock_prices.shift(MOMENTUM_LOOKBACK))
    .sub(1)
)

annualised_volatility_3m = (
    stock_daily_returns
    .rolling(
        window=VOLATILITY_LOOKBACK,
        min_periods=VOLATILITY_LOOKBACK,
    )
    .std(ddof=1)
    .mul(np.sqrt(TRADING_DAYS))
)

benchmark_series = (
    benchmark_prices[BENCHMARK_NAME]
    .copy()
)

benchmark_ma_200 = (
    benchmark_series
    .rolling(
        window=TREND_LOOKBACK,
        min_periods=TREND_LOOKBACK,
    )
    .mean()
)

# Last available trading session in each calendar month
month_end_dates = (
    stock_prices
    .groupby(stock_prices.index.to_period("M"))
    .tail(1)
    .index
)

valid_rebalance_dates = []

for rebalance_date in month_end_dates:

    momentum_available = (
        momentum_6m
        .loc[rebalance_date, growth_tickers]
        .notna()
        .all()
    )

    volatility_available = (
        annualised_volatility_3m
        .loc[rebalance_date, portfolio_tickers]
        .notna()
        .all()
    )

    trend_available = pd.notna(
        benchmark_ma_200.loc[rebalance_date]
    )

    if (
        momentum_available
        and volatility_available
        and trend_available
    ):
        valid_rebalance_dates.append(rebalance_date)

valid_rebalance_dates = pd.DatetimeIndex(
    valid_rebalance_dates
)

if len(valid_rebalance_dates) == 0:

    st.error(
        "No valid strategy rebalance dates were generated. "
        "Choose an earlier historical start date."
    )

    st.stop()


# ---------------------------------------------------------
# Adaptive target-weight function
# ---------------------------------------------------------

def calculate_target_weights(rebalance_date):

    # Resilient Compounders: inverse-volatility weights
    resilient_volatility = (
        annualised_volatility_3m
        .loc[rebalance_date, resilient_tickers]
        .astype(float)
    )

    inverse_volatility = (
        1 / resilient_volatility.replace(0, np.nan)
    )

    resilient_weights = (
        inverse_volatility
        / inverse_volatility.sum()
    )

    # Growth Leaders: momentum and low-volatility scoring
    growth_momentum = (
        momentum_6m
        .loc[rebalance_date, growth_tickers]
        .astype(float)
    )

    growth_volatility = (
        annualised_volatility_3m
        .loc[rebalance_date, growth_tickers]
        .astype(float)
    )

    momentum_score = growth_momentum.rank(
        pct=True,
        method="average",
        ascending=True,
    )

    low_volatility_score = growth_volatility.rank(
        pct=True,
        method="average",
        ascending=False,
    )

    growth_combined_score = (
        0.70 * momentum_score
        + 0.30 * low_volatility_score
    )

    growth_weights = (
        growth_combined_score
        / growth_combined_score.sum()
    )

    # Combine both buckets
    equity_weights = pd.Series(
        0.0,
        index=portfolio_tickers,
        dtype=float,
    )

    equity_weights.loc[resilient_tickers] = (
        0.50 * resilient_weights
    )

    equity_weights.loc[growth_tickers] = (
        0.50 * growth_weights
    )

    # Market-risk overlay
    nifty_level = float(
        benchmark_series.loc[rebalance_date]
    )

    nifty_ma_200 = float(
        benchmark_ma_200.loc[rebalance_date]
    )

    if nifty_level >= nifty_ma_200:

        market_regime = "Normal"
        equity_exposure = FULL_EQUITY_EXPOSURE

    else:

        market_regime = "Defensive"
        equity_exposure = DEFENSIVE_EQUITY_EXPOSURE

    final_stock_weights = (
        equity_weights * equity_exposure
    )

    cash_weight = (
        1.0 - final_stock_weights.sum()
    )

    return {
        "Stock Weights": final_stock_weights,
        "Cash Weight": cash_weight,
        "Equity Exposure": equity_exposure,
        "Market Regime": market_regime,
        "Nifty 50": nifty_level,
        "Nifty 200-Day MA": nifty_ma_200,
    }


# ---------------------------------------------------------
# Generate historical target weights
# ---------------------------------------------------------

target_weight_records = []
regime_records = []

for rebalance_date in valid_rebalance_dates:

    strategy_result = calculate_target_weights(
        rebalance_date
    )

    weight_row = strategy_result[
        "Stock Weights"
    ].copy()

    weight_row["Cash"] = strategy_result[
        "Cash Weight"
    ]

    weight_row.name = rebalance_date

    target_weight_records.append(weight_row)

    regime_records.append(
        {
            "Date": rebalance_date,
            "Market Regime": strategy_result[
                "Market Regime"
            ],
            "Equity Exposure": strategy_result[
                "Equity Exposure"
            ],
            "Cash Weight": strategy_result[
                "Cash Weight"
            ],
            "Nifty 50": strategy_result[
                "Nifty 50"
            ],
            "Nifty 200-Day MA": strategy_result[
                "Nifty 200-Day MA"
            ],
        }
    )

strategy_target_weights = (
    pd.DataFrame(target_weight_records)
    .sort_index()
)

strategy_regimes = (
    pd.DataFrame(regime_records)
    .set_index("Date")
    .sort_index()
)

if not np.allclose(
    strategy_target_weights.sum(axis=1),
    1.0,
):

    st.error(
        "Strategy target weights do not total 100%."
    )

    st.stop()


# Latest available recommendation
latest_signal_date = (
    strategy_target_weights.index[-1]
)

latest_target_weights = (
    strategy_target_weights
    .loc[latest_signal_date]
)

latest_market_regime = (
    strategy_regimes
    .loc[latest_signal_date, "Market Regime"]
)

latest_equity_exposure = (
    strategy_regimes
    .loc[latest_signal_date, "Equity Exposure"]
)

latest_cash_weight = (
    strategy_regimes
    .loc[latest_signal_date, "Cash Weight"]
)


# ---------------------------------------------------------
# Lagged execution schedule
# ---------------------------------------------------------

execution_records = []
execution_schedule_records = []

for signal_date, target_row in strategy_target_weights.iterrows():

    next_position = stock_prices.index.searchsorted(
        signal_date,
        side="right",
    )

    # Skip the latest signal if no subsequent trading day exists
    if next_position >= len(stock_prices.index):
        continue

    execution_date = stock_prices.index[next_position]

    execution_row = target_row.copy()
    execution_row.name = execution_date

    execution_records.append(execution_row)

    execution_schedule_records.append(
        {
            "Signal Date": signal_date,
            "Execution Date": execution_date,
            "Market Regime": strategy_regimes.loc[
                signal_date,
                "Market Regime",
            ],
            "Equity Exposure": strategy_regimes.loc[
                signal_date,
                "Equity Exposure",
            ],
            "Cash Weight": target_row["Cash"],
        }
    )


execution_targets = (
    pd.DataFrame(execution_records)
    .groupby(level=0)
    .last()
    .sort_index()
)

execution_schedule = (
    pd.DataFrame(execution_schedule_records)
    .drop_duplicates(
        subset="Execution Date",
        keep="last",
    )
    .set_index("Execution Date")
    .sort_index()
)


if execution_targets.empty:

    st.error(
        "No executable strategy allocations were generated."
    )

    st.stop()


if not np.allclose(
    execution_targets.sum(axis=1),
    1.0,
):

    st.error(
        "Execution weights do not total 100%."
    )

    st.stop()


# ---------------------------------------------------------
# Adaptive strategy backtest
# ---------------------------------------------------------

daily_cash_return = (
    (1 + RISK_FREE_RATE)
    ** (1 / TRADING_DAYS)
    - 1
)

backtest_stock_returns = (
    stock_prices
    .pct_change()
)

first_execution_date = (
    execution_targets.index.min()
)

first_execution_position = (
    stock_prices.index.get_loc(
        first_execution_date
    )
)

if first_execution_position == 0:

    st.error(
        "A prior trading date is required to "
        "initialise the strategy backtest."
    )

    st.stop()


backtest_initial_date = (
    stock_prices.index[
        first_execution_position - 1
    ]
)

simulation_dates = (
    stock_prices
    .loc[first_execution_date:]
    .index
)

asset_names = portfolio_tickers + ["Cash"]

current_asset_values = pd.Series(
    0.0,
    index=asset_names,
    dtype=float,
)

current_asset_values["Cash"] = (
    float(initial_capital)
)

strategy_value_records = {
    backtest_initial_date: float(initial_capital)
}

weight_history_records = {
    backtest_initial_date:
        current_asset_values
        / float(initial_capital)
}

turnover_records = {}
transaction_cost_records = {}


for current_date in simulation_dates:

    portfolio_value_before_trade = (
        current_asset_values.sum()
    )

    # Rebalance at the beginning of an execution date
    if current_date in execution_targets.index:

        target_weights = (
            execution_targets
            .loc[current_date]
            .reindex(asset_names)
            .astype(float)
        )

        current_weights = (
            current_asset_values
            / portfolio_value_before_trade
        )

        one_way_turnover = (
            0.5
            * (
                target_weights
                - current_weights
            )
            .abs()
            .sum()
        )

        transaction_cost = (
            portfolio_value_before_trade
            * one_way_turnover
            * TRANSACTION_COST_RATE
        )

        portfolio_value_after_cost = (
            portfolio_value_before_trade
            - transaction_cost
        )

        current_asset_values = (
            target_weights
            * portfolio_value_after_cost
        )

        turnover_records[current_date] = (
            one_way_turnover
        )

        transaction_cost_records[current_date] = (
            transaction_cost
        )

    # Apply the daily equity returns
    equity_returns_today = (
        backtest_stock_returns
        .loc[current_date, portfolio_tickers]
        .fillna(0.0)
    )

    current_asset_values.loc[
        portfolio_tickers
    ] *= (
        1 + equity_returns_today
    )

    # Apply the daily cash return
    current_asset_values["Cash"] *= (
        1 + daily_cash_return
    )

    current_portfolio_value = (
        current_asset_values.sum()
    )

    strategy_value_records[current_date] = (
        current_portfolio_value
    )

    weight_history_records[current_date] = (
        current_asset_values
        / current_portfolio_value
    )


adaptive_strategy_value = (
    pd.Series(
        strategy_value_records,
        name="Adaptive Barbell Strategy",
    )
    .sort_index()
)

strategy_weight_history = (
    pd.DataFrame(weight_history_records)
    .T
    .sort_index()
)

strategy_turnover = (
    pd.Series(
        turnover_records,
        name="One-Way Turnover",
        dtype=float,
    )
    .sort_index()
)

strategy_transaction_costs = (
    pd.Series(
        transaction_cost_records,
        name="Transaction Cost",
        dtype=float,
    )
    .sort_index()
)


if (
    adaptive_strategy_value.isna().any()
    or (adaptive_strategy_value <= 0).any()
):

    st.error(
        "Invalid values were generated by the strategy backtest."
    )

    st.stop()


# ---------------------------------------------------------
# Fixed India 10 comparison portfolio
# ---------------------------------------------------------

comparison_stock_prices = (
    stock_prices
    .loc[backtest_initial_date:]
    .copy()
)

initial_comparison_prices = (
    comparison_stock_prices
    .loc[backtest_initial_date]
)

equal_weights = (
    portfolio
    .set_index("Ticker")["Initial Weight"]
    .reindex(portfolio_tickers)
)

fixed_initial_allocations = (
    equal_weights
    * float(initial_capital)
)

fixed_shares = (
    fixed_initial_allocations
    / initial_comparison_prices
)

fixed_portfolio_value = (
    comparison_stock_prices
    .mul(fixed_shares, axis=1)
    .sum(axis=1)
    .rename("Fixed India 10 Portfolio")
)


# ---------------------------------------------------------
# Nifty 50 comparison investment
# ---------------------------------------------------------

comparison_benchmark_prices = (
    benchmark_prices[BENCHMARK_NAME]
    .loc[backtest_initial_date:]
)

benchmark_units = (
    float(initial_capital)
    / comparison_benchmark_prices.iloc[0]
)

nifty_value = (
    comparison_benchmark_prices
    * benchmark_units
)

nifty_value.name = BENCHMARK_NAME


# ---------------------------------------------------------
# Combine comparable portfolio-value series
# ---------------------------------------------------------

comparison_values = pd.concat(
    [
        adaptive_strategy_value,
        fixed_portfolio_value,
        nifty_value,
    ],
    axis=1,
).dropna()


if comparison_values.empty:

    st.error(
        "Comparable investment series could not be created."
    )

    st.stop()


if not np.allclose(
    comparison_values.iloc[0].astype(float),
    float(initial_capital),
):

    st.error(
        "The comparison portfolios do not begin "
        "with the same initial capital."
    )

    st.stop()


# ---------------------------------------------------------
# Performance-metric function
# ---------------------------------------------------------

def calculate_performance_metrics(
    value_series,
    benchmark_returns,
):

    value_series = (
        value_series
        .dropna()
        .astype(float)
    )

    return_series = (
        value_series
        .pct_change()
        .dropna()
    )

    aligned_returns = pd.concat(
        [
            return_series.rename("Asset"),
            benchmark_returns.rename("Benchmark"),
        ],
        axis=1,
    ).dropna()

    years = (
        value_series.index[-1]
        - value_series.index[0]
    ).days / 365.25

    total_return = (
        value_series.iloc[-1]
        / value_series.iloc[0]
        - 1
    )

    cagr = (
        value_series.iloc[-1]
        / value_series.iloc[0]
    ) ** (1 / years) - 1

    annualised_volatility = (
        return_series.std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )

    daily_risk_free_rate = (
        (1 + RISK_FREE_RATE)
        ** (1 / TRADING_DAYS)
        - 1
    )

    excess_returns = (
        return_series
        - daily_risk_free_rate
    )

    annualised_excess_return = (
        excess_returns.mean()
        * TRADING_DAYS
    )

    sharpe_ratio = (
        annualised_excess_return
        / annualised_volatility
        if annualised_volatility > 0
        else np.nan
    )

    downside_returns = (
        excess_returns
        .clip(upper=0.0)
    )

    downside_deviation = (
        np.sqrt(
            np.mean(
                downside_returns ** 2
            )
        )
        * np.sqrt(TRADING_DAYS)
    )

    sortino_ratio = (
        annualised_excess_return
        / downside_deviation
        if downside_deviation > 0
        else np.nan
    )

    drawdown = (
        value_series
        / value_series.cummax()
        - 1
    )

    maximum_drawdown = (
        drawdown.min()
    )

    calmar_ratio = (
        cagr / abs(maximum_drawdown)
        if maximum_drawdown < 0
        else np.nan
    )

    benchmark_variance = (
        aligned_returns["Benchmark"]
        .var(ddof=1)
    )

    beta = (
        aligned_returns
        .cov()
        .loc["Asset", "Benchmark"]
        / benchmark_variance
        if benchmark_variance > 0
        else np.nan
    )

    correlation = (
        aligned_returns["Asset"]
        .corr(
            aligned_returns["Benchmark"]
        )
    )

    return {
        "Ending Value (₹)": value_series.iloc[-1],
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualised Volatility": annualised_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Maximum Drawdown": maximum_drawdown,
        "Calmar Ratio": calmar_ratio,
        "Beta vs Nifty 50": beta,
        "Correlation vs Nifty 50": correlation,
    }


comparison_daily_returns = (
    comparison_values
    .pct_change()
    .dropna()
)

benchmark_return_series = (
    comparison_daily_returns[
        BENCHMARK_NAME
    ]
)

metric_results = {}

for investment_name in comparison_values.columns:

    metric_results[investment_name] = (
        calculate_performance_metrics(
            value_series=comparison_values[
                investment_name
            ],
            benchmark_returns=benchmark_return_series,
        )
    )

performance_table = pd.DataFrame(
    metric_results
)

comparison_drawdowns = (
    comparison_values
    .div(comparison_values.cummax())
    .sub(1)
)


# ---------------------------------------------------------
# Main strategy dashboard
# ---------------------------------------------------------

adaptive_metrics = performance_table[
    "Adaptive Barbell Strategy"
]

fixed_metrics = performance_table[
    "Fixed India 10 Portfolio"
]

nifty_metrics = performance_table[
    BENCHMARK_NAME
]


# ---------------------------------------------------------
# Latest strategy status
# ---------------------------------------------------------

st.markdown("## Current Strategy Status")

status_col_1, status_col_2, status_col_3, status_col_4 = (
    st.columns(4)
)

status_col_1.metric(
    "Latest Signal Date",
    latest_signal_date.strftime("%d %b %Y"),
)

status_col_2.metric(
    "Market Regime",
    latest_market_regime,
)

status_col_3.metric(
    "Equity Exposure",
    f"{latest_equity_exposure:.0%}",
)

status_col_4.metric(
    "Cash Allocation",
    f"{latest_cash_weight:.0%}",
)


# ---------------------------------------------------------
# Key performance indicators
# ---------------------------------------------------------

st.markdown("## Adaptive Strategy Performance")

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = (
    st.columns(4)
)

metric_col_1.metric(
    "Ending Portfolio Value",
    f"₹{adaptive_metrics['Ending Value (₹)']:,.0f}",
)

metric_col_2.metric(
    "Strategy CAGR",
    f"{adaptive_metrics['CAGR']:.2%}",
    delta=(
        f"{adaptive_metrics['CAGR'] - fixed_metrics['CAGR']:+.2%} "
        "vs fixed"
    ),
)

metric_col_3.metric(
    "Annualised Volatility",
    f"{adaptive_metrics['Annualised Volatility']:.2%}",
    delta=(
        f"{adaptive_metrics['Annualised Volatility'] - fixed_metrics['Annualised Volatility']:+.2%} "
        "vs fixed"
    ),
    delta_color="inverse",
)

metric_col_4.metric(
    "Maximum Drawdown",
    f"{adaptive_metrics['Maximum Drawdown']:.2%}",
    delta=(
        f"{adaptive_metrics['Maximum Drawdown'] - fixed_metrics['Maximum Drawdown']:+.2%} "
        "vs fixed"
    ),
)


# ---------------------------------------------------------
# Portfolio-value comparison chart
# ---------------------------------------------------------

st.markdown("## Growth of Investment")

performance_figure, performance_axis = plt.subplots(
    figsize=(13, 6)
)

for column in comparison_values.columns:

    performance_axis.plot(
        comparison_values.index,
        comparison_values[column],
        linewidth=2.2,
        label=column,
    )

performance_axis.axhline(
    float(initial_capital),
    linewidth=1,
    linestyle=":",
    label="Initial Capital",
)

performance_axis.set_title(
    f"Growth of ₹{float(initial_capital):,.0f}"
)

performance_axis.set_xlabel("Date")
performance_axis.set_ylabel("Portfolio Value (₹)")
performance_axis.legend()
performance_axis.grid(alpha=0.25)

performance_figure.tight_layout()

st.pyplot(
    performance_figure,
    use_container_width=True,
)

plt.close(performance_figure)


# ---------------------------------------------------------
# Performance-comparison table
# ---------------------------------------------------------

st.markdown("## Risk and Return Comparison")

selected_metrics = performance_table.loc[
    [
        "Ending Value (₹)",
        "Total Return",
        "CAGR",
        "Annualised Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Maximum Drawdown",
        "Calmar Ratio",
        "Beta vs Nifty 50",
        "Correlation vs Nifty 50",
    ]
].copy()

# Convert to object before inserting formatted strings.
# This avoids Pandas 3.x dtype errors.
formatted_metrics = (
    selected_metrics
    .copy()
    .astype(object)
)

for column in formatted_metrics.columns:

    formatted_metrics.loc[
        "Ending Value (₹)",
        column,
    ] = (
        f"₹{selected_metrics.loc['Ending Value (₹)', column]:,.0f}"
    )

    for metric in [
        "Total Return",
        "CAGR",
        "Annualised Volatility",
        "Maximum Drawdown",
    ]:

        formatted_metrics.loc[
            metric,
            column,
        ] = (
            f"{selected_metrics.loc[metric, column]:.2%}"
        )

    for metric in [
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Beta vs Nifty 50",
        "Correlation vs Nifty 50",
    ]:

        formatted_metrics.loc[
            metric,
            column,
        ] = (
            f"{selected_metrics.loc[metric, column]:.2f}"
        )

st.dataframe(
    formatted_metrics,
    use_container_width=True,
)


# ---------------------------------------------------------
# Drawdown comparison
# ---------------------------------------------------------

st.markdown("## Historical Drawdown")

drawdown_figure, drawdown_axis = plt.subplots(
    figsize=(13, 5)
)

for column in comparison_drawdowns.columns:

    drawdown_axis.plot(
        comparison_drawdowns.index,
        comparison_drawdowns[column],
        linewidth=2,
        label=column,
    )

drawdown_axis.axhline(
    0,
    linewidth=1,
)

drawdown_axis.set_xlabel("Date")
drawdown_axis.set_ylabel("Drawdown")

drawdown_axis.yaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda value, position: f"{value:.0%}"
    )
)

drawdown_axis.legend()
drawdown_axis.grid(alpha=0.25)

drawdown_figure.tight_layout()

st.pyplot(
    drawdown_figure,
    use_container_width=True,
)

plt.close(drawdown_figure)


# ---------------------------------------------------------
# Preliminary interpretation
# ---------------------------------------------------------

st.markdown("## Preliminary Interpretation")

cagr_difference = (
    adaptive_metrics["CAGR"]
    - fixed_metrics["CAGR"]
)

volatility_difference = (
    adaptive_metrics["Annualised Volatility"]
    - fixed_metrics["Annualised Volatility"]
)

drawdown_improvement = (
    adaptive_metrics["Maximum Drawdown"]
    - fixed_metrics["Maximum Drawdown"]
)

st.markdown(
    f'''
    The Adaptive Barbell strategy produced a CAGR of
    **{adaptive_metrics["CAGR"]:.2%}**, compared with
    **{fixed_metrics["CAGR"]:.2%}** for the fixed portfolio.

    It reduced annualised volatility by
    **{abs(volatility_difference):.2%}** and improved maximum
    drawdown by **{drawdown_improvement:.2%}**.

    The strategy's Sharpe ratio was
    **{adaptive_metrics["Sharpe Ratio"]:.2f}**, compared with
    **{fixed_metrics["Sharpe Ratio"]:.2f}** for the fixed portfolio.

    The strategy therefore generated a lower absolute return during
    the test period but achieved stronger downside protection and
    risk-adjusted performance.
    '''
)


# ---------------------------------------------------------
# Methodology disclaimer
# ---------------------------------------------------------

st.warning(
    "Historical results do not guarantee future performance. "
    "The ten-stock universe was selected using present-day knowledge "
    "and remains exposed to selection and survivorship bias."
)


# ---------------------------------------------------------
# v0.3 robustness dashboard
# ---------------------------------------------------------

st.markdown("---")
st.markdown("# Strategy Robustness")

final_assessment = robustness_data[
    "final_robustness_assessment"
].copy()

out_of_sample_performance = robustness_data[
    "out_of_sample_performance"
].copy()

parameter_sensitivity = robustness_data[
    "parameter_sensitivity"
].copy()


def assessment_result(metric_name):
    """
    Read a result from the exported final assessment table.
    """

    return str(
        final_assessment.loc[
            metric_name,
            "Result",
        ]
    )


robust_col_1, robust_col_2, robust_col_3, robust_col_4 = (
    st.columns(4)
)

robust_col_1.metric(
    "Raw Robustness Score",
    assessment_result(
        "Raw Robustness Score"
    ),
)

robust_col_2.metric(
    "Adjusted Confidence Score",
    assessment_result(
        "Adjusted Research-Confidence Score"
    ),
)

robust_col_3.metric(
    "Confidence Classification",
    assessment_result(
        "Confidence Classification"
    ),
)

robust_col_4.metric(
    "Parameter Combinations",
    f"{len(parameter_sensitivity):,}",
)


st.markdown("## Out-of-Sample Performance")

st.caption(
    "Performance from 1 January 2022 onward. "
    "Each series was rebased to ₹10 lakh at the beginning "
    "of the out-of-sample period."
)

oos_metrics_to_display = [
    "Total Return",
    "CAGR",
    "Annualised Volatility",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Maximum Drawdown",
    "Calmar Ratio",
    "Beta vs Nifty 50",
]

formatted_oos_performance = (
    out_of_sample_performance
    .loc[oos_metrics_to_display]
    .copy()
    .astype(object)
)

percentage_metrics = [
    "Total Return",
    "CAGR",
    "Annualised Volatility",
    "Maximum Drawdown",
]

ratio_metrics = [
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
    "Beta vs Nifty 50",
]

for column in formatted_oos_performance.columns:

    for metric in percentage_metrics:

        formatted_oos_performance.loc[
            metric,
            column,
        ] = (
            f"{out_of_sample_performance.loc[metric, column]:.2%}"
        )

    for metric in ratio_metrics:

        formatted_oos_performance.loc[
            metric,
            column,
        ] = (
            f"{out_of_sample_performance.loc[metric, column]:.2f}"
        )


st.dataframe(
    formatted_oos_performance,
    use_container_width=True,
)


adaptive_oos = out_of_sample_performance[
    "Adaptive Barbell Strategy"
]

fixed_oos = out_of_sample_performance[
    "Fixed India 10 Portfolio"
]

st.markdown(
    f"""
    During the out-of-sample period, the Adaptive strategy generated
    a CAGR of **{adaptive_oos["CAGR"]:.2%}**, compared with
    **{fixed_oos["CAGR"]:.2%}** for the fixed portfolio.

    Annualised volatility declined from
    **{fixed_oos["Annualised Volatility"]:.2%}** to
    **{adaptive_oos["Annualised Volatility"]:.2%}**, while maximum
    drawdown improved from **{fixed_oos["Maximum Drawdown"]:.2%}**
    to **{adaptive_oos["Maximum Drawdown"]:.2%}**.

    The strategy therefore preserved most of the fixed portfolio's
    return while delivering materially stronger downside protection.
    """
)

# ---------------------------------------------------------
# v0.3 detailed robustness analysis
# ---------------------------------------------------------

st.markdown("## Detailed Robustness Analysis")

(
    parameter_tab,
    cost_tab,
    rolling_tab,
    regime_tab,
    scorecard_tab,
) = st.tabs(
    [
        "Parameter Stability",
        "Transaction Costs",
        "Rolling Performance",
        "Market Regimes",
        "Confidence Scorecard",
    ]
)


# =========================================================
# TAB 1 — Parameter stability
# =========================================================

with parameter_tab:

    sensitivity_data = robustness_data[
        "parameter_sensitivity"
    ].copy()

    parameter_col_1, parameter_col_2, parameter_col_3, parameter_col_4 = (
        st.columns(4)
    )

    parameter_col_1.metric(
        "Combinations Tested",
        f"{len(sensitivity_data):,}",
    )

    parameter_col_2.metric(
        "Median CAGR",
        f"{sensitivity_data['CAGR'].median():.2%}",
    )

    parameter_col_3.metric(
        "Minimum Sharpe",
        f"{sensitivity_data['Sharpe Ratio'].min():.2f}",
    )

    parameter_col_4.metric(
        "Median Drawdown",
        (
            f"{sensitivity_data['Maximum Drawdown'].median():.2%}"
        ),
    )

    parameter_scatter_figure, parameter_scatter_axis = (
        plt.subplots(figsize=(11, 6))
    )

    parameter_scatter_axis.scatter(
        sensitivity_data[
            "Annualised Volatility"
        ] * 100,
        sensitivity_data["CAGR"] * 100,
        alpha=0.65,
        label="Alternative parameter combinations",
    )

    base_parameter_row = sensitivity_data.loc[
        (
            sensitivity_data["Momentum Lookback"] == 126
        )
        & (
            sensitivity_data["Volatility Lookback"] == 63
        )
        & (
            sensitivity_data["Trend Lookback"] == 200
        )
        & (
            np.isclose(
                sensitivity_data["Defensive Exposure"],
                0.70,
            )
        )
    ]

    if not base_parameter_row.empty:

        parameter_scatter_axis.scatter(
            base_parameter_row[
                "Annualised Volatility"
            ] * 100,
            base_parameter_row["CAGR"] * 100,
            marker="X",
            s=160,
            label="Frozen base parameters",
        )

    parameter_scatter_axis.set_title(
        "Out-of-Sample Parameter Stability"
    )

    parameter_scatter_axis.set_xlabel(
        "Annualised Volatility (%)"
    )

    parameter_scatter_axis.set_ylabel(
        "CAGR (%)"
    )

    parameter_scatter_axis.grid(alpha=0.25)
    parameter_scatter_axis.legend()
    parameter_scatter_figure.tight_layout()

    st.pyplot(
        parameter_scatter_figure,
        use_container_width=True,
    )

    plt.close(parameter_scatter_figure)

    st.markdown(
        f"""
        Across **{len(sensitivity_data)} parameter combinations**,
        CAGR ranged from
        **{sensitivity_data["CAGR"].min():.2%}** to
        **{sensitivity_data["CAGR"].max():.2%}**.

        The median Sharpe ratio was
        **{sensitivity_data["Sharpe Ratio"].median():.2f}**,
        while the weakest Sharpe ratio was still
        **{sensitivity_data["Sharpe Ratio"].min():.2f}**.

        This indicates that the strategy's results were not dependent
        on one exact momentum, volatility, trend or defensive-exposure
        assumption.
        """
    )


# =========================================================
# TAB 2 — Transaction-cost stress testing
# =========================================================

with cost_tab:

    cost_data = robustness_data[
        "transaction_cost_stress"
    ].copy()

    highest_cost_row = (
        cost_data
        .sort_values("Transaction Cost Rate")
        .iloc[-1]
    )

    cost_col_1, cost_col_2, cost_col_3, cost_col_4 = (
        st.columns(4)
    )

    cost_col_1.metric(
        "Highest Tested Cost",
        (
            f"{highest_cost_row['Transaction Cost Rate']:.2%}"
        ),
    )

    cost_col_2.metric(
        "CAGR at Highest Cost",
        f"{highest_cost_row['CAGR']:.2%}",
    )

    cost_col_3.metric(
        "Sharpe at Highest Cost",
        f"{highest_cost_row['Sharpe Ratio']:.2f}",
    )

    cost_col_4.metric(
        "CAGR Drag",
        (
            f"{highest_cost_row['CAGR Drag vs Zero Cost']:.2%}"
        ),
    )

    cost_figure, cost_axis = plt.subplots(
        figsize=(11, 6)
    )

    cost_axis.plot(
        cost_data["Transaction Cost Rate"] * 100,
        cost_data["CAGR"] * 100,
        marker="o",
        linewidth=2,
    )

    cost_axis.set_title(
        "Out-of-Sample CAGR under Higher Transaction Costs"
    )

    cost_axis.set_xlabel(
        "Transaction Cost per Unit of Turnover (%)"
    )

    cost_axis.set_ylabel("CAGR (%)")
    cost_axis.grid(alpha=0.25)
    cost_figure.tight_layout()

    st.pyplot(
        cost_figure,
        use_container_width=True,
    )

    plt.close(cost_figure)

    formatted_cost_data = cost_data[
        [
            "Transaction Cost Rate",
            "Ending Value (₹)",
            "CAGR",
            "Sharpe Ratio",
            "Maximum Drawdown",
            "Calmar Ratio",
            "CAGR Drag vs Zero Cost",
        ]
    ].copy().astype(object)

    for row_number in formatted_cost_data.index:

        formatted_cost_data.loc[
            row_number,
            "Transaction Cost Rate",
        ] = (
            f"{cost_data.loc[row_number, 'Transaction Cost Rate']:.2%}"
        )

        formatted_cost_data.loc[
            row_number,
            "Ending Value (₹)",
        ] = (
            f"₹{cost_data.loc[row_number, 'Ending Value (₹)']:,.0f}"
        )

        for percentage_metric in [
            "CAGR",
            "Maximum Drawdown",
            "CAGR Drag vs Zero Cost",
        ]:

            formatted_cost_data.loc[
                row_number,
                percentage_metric,
            ] = (
                f"{cost_data.loc[row_number, percentage_metric]:.2%}"
            )

        for ratio_metric in [
            "Sharpe Ratio",
            "Calmar Ratio",
        ]:

            formatted_cost_data.loc[
                row_number,
                ratio_metric,
            ] = (
                f"{cost_data.loc[row_number, ratio_metric]:.2f}"
            )

    st.dataframe(
        formatted_cost_data,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        Performance deteriorated gradually rather than collapsing as
        transaction costs increased. The strategy remained economically
        viable even under the highest tested cost assumption.
        """
    )


# =========================================================
# TAB 3 — Rolling performance
# =========================================================

with rolling_tab:

    rolling_sharpe_data = robustness_data[
        "rolling_sharpe"
    ].copy()

    rolling_volatility_data = robustness_data[
        "rolling_volatility"
    ].copy()

    rolling_drawdown_data = robustness_data[
        "rolling_drawdown"
    ].copy()

    rolling_beta_data = robustness_data[
        "rolling_beta"
    ].copy()

    rolling_oos_start = pd.Timestamp(
        "2022-01-01"
    )

    adaptive_label = (
        "Adaptive Barbell Strategy"
    )

    fixed_label = (
        "Fixed India 10 Portfolio"
    )

    rolling_sharpe_oos = (
        rolling_sharpe_data
        .loc[
            rolling_sharpe_data.index
            >= rolling_oos_start
        ]
        .dropna()
    )

    rolling_volatility_oos = (
        rolling_volatility_data
        .loc[
            rolling_volatility_data.index
            >= rolling_oos_start
        ]
        .dropna()
    )

    rolling_drawdown_oos = (
        rolling_drawdown_data
        .loc[
            rolling_drawdown_data.index
            >= rolling_oos_start
        ]
        .dropna()
    )

    rolling_beta_oos = (
        rolling_beta_data
        .loc[
            rolling_beta_data.index
            >= rolling_oos_start
        ]
        .dropna()
    )

    adaptive_sharpe_median = (
        rolling_sharpe_oos[
            adaptive_label
        ].median()
    )

    fixed_sharpe_median = (
        rolling_sharpe_oos[
            fixed_label
        ].median()
    )

    lower_volatility_frequency = (
        rolling_volatility_oos[
            adaptive_label
        ]
        < rolling_volatility_oos[
            fixed_label
        ]
    ).mean()

    better_drawdown_frequency = (
        rolling_drawdown_oos[
            adaptive_label
        ]
        > rolling_drawdown_oos[
            fixed_label
        ]
    ).mean()

    adaptive_beta_median = (
        rolling_beta_oos[
            adaptive_label
        ].median()
    )

    rolling_col_1, rolling_col_2, rolling_col_3, rolling_col_4 = (
        st.columns(4)
    )

    rolling_col_1.metric(
        "Median Adaptive Sharpe",
        f"{adaptive_sharpe_median:.2f}",
    )

    rolling_col_2.metric(
        "Median Fixed Sharpe",
        f"{fixed_sharpe_median:.2f}",
    )

    rolling_col_3.metric(
        "Lower Volatility Periods",
        f"{lower_volatility_frequency:.2%}",
    )

    rolling_col_4.metric(
        "Better Drawdown Periods",
        f"{better_drawdown_frequency:.2%}",
    )

    rolling_figure, rolling_axis = plt.subplots(
        figsize=(12, 6)
    )

    for investment_name in [
        adaptive_label,
        fixed_label,
        BENCHMARK_NAME,
    ]:

        rolling_axis.plot(
            rolling_sharpe_oos.index,
            rolling_sharpe_oos[
                investment_name
            ],
            linewidth=2,
            label=investment_name,
        )

    rolling_axis.axhline(
        0,
        linewidth=1,
    )

    rolling_axis.set_title(
        "Out-of-Sample Rolling One-Year Sharpe Ratio"
    )

    rolling_axis.set_xlabel("Date")
    rolling_axis.set_ylabel("Rolling Sharpe Ratio")
    rolling_axis.grid(alpha=0.25)
    rolling_axis.legend()
    rolling_figure.tight_layout()

    st.pyplot(
        rolling_figure,
        use_container_width=True,
    )

    plt.close(rolling_figure)

    st.markdown(
        f"""
        The Adaptive strategy generated lower rolling volatility in
        **{lower_volatility_frequency:.2%}** of out-of-sample periods
        and better rolling drawdowns in
        **{better_drawdown_frequency:.2%}** of periods.

        Its median rolling beta was
        **{adaptive_beta_median:.2f}**.

        The most consistent evidence therefore relates to risk
        reduction rather than short-term return maximisation.
        """
    )


# =========================================================
# TAB 4 — Market-regime analysis
# =========================================================

with regime_tab:

    regime_data = robustness_data[
        "market_regime_performance"
    ].copy()

    regime_display = regime_data[
        [
            "Market Regime",
            "Investment",
            "Annualised Return",
            "Annualised Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
        ]
    ].copy().astype(object)

    for row_number in regime_display.index:

        for percentage_metric in [
            "Annualised Return",
            "Annualised Volatility",
            "Maximum Drawdown",
        ]:

            regime_display.loc[
                row_number,
                percentage_metric,
            ] = (
                f"{regime_data.loc[row_number, percentage_metric]:.2%}"
            )

        regime_display.loc[
            row_number,
            "Sharpe Ratio",
        ] = (
            f"{regime_data.loc[row_number, 'Sharpe Ratio']:.2f}"
        )

    st.dataframe(
        regime_display,
        use_container_width=True,
        hide_index=True,
    )

    defensive_regime_data = (
        regime_data.loc[
            regime_data["Market Regime"]
            == "Defensive"
        ]
        .set_index("Investment")
    )

    adaptive_defensive = defensive_regime_data.loc[
        adaptive_label
    ]

    fixed_defensive = defensive_regime_data.loc[
        fixed_label
    ]

    regime_col_1, regime_col_2, regime_col_3, regime_col_4 = (
        st.columns(4)
    )

    regime_col_1.metric(
        "Adaptive Defensive Return",
        (
            f"{adaptive_defensive['Annualised Return']:.2%}"
        ),
    )

    regime_col_2.metric(
        "Fixed Defensive Return",
        (
            f"{fixed_defensive['Annualised Return']:.2%}"
        ),
    )

    regime_col_3.metric(
        "Adaptive Defensive Volatility",
        (
            f"{adaptive_defensive['Annualised Volatility']:.2%}"
        ),
    )

    regime_col_4.metric(
        "Adaptive Defensive Drawdown",
        (
            f"{adaptive_defensive['Maximum Drawdown']:.2%}"
        ),
    )

    st.markdown(
        f"""
        During defensive regimes, the Adaptive strategy produced an
        annualised return of
        **{adaptive_defensive["Annualised Return"]:.2%}**, compared
        with **{fixed_defensive["Annualised Return"]:.2%}** for the
        fixed portfolio.

        Volatility fell from
        **{fixed_defensive["Annualised Volatility"]:.2%}** to
        **{adaptive_defensive["Annualised Volatility"]:.2%}**, while
        maximum drawdown improved from
        **{fixed_defensive["Maximum Drawdown"]:.2%}** to
        **{adaptive_defensive["Maximum Drawdown"]:.2%}**.
        """
    )


# =========================================================
# TAB 5 — Confidence scorecard
# =========================================================

with scorecard_tab:

    scorecard_data = robustness_data[
        "robustness_scorecard"
    ].copy()

    limitations_data = robustness_data[
        "research_limitations"
    ].copy()

    assessment_data = robustness_data[
        "final_robustness_assessment"
    ].copy()

    formatted_scorecard = scorecard_data.copy()

    formatted_scorecard[
        "Maximum Points"
    ] = formatted_scorecard[
        "Maximum Points"
    ].map(
        lambda value: f"{value:.0f}"
    )

    formatted_scorecard[
        "Points Awarded"
    ] = formatted_scorecard[
        "Points Awarded"
    ].map(
        lambda value: f"{value:.2f}"
    )

    st.markdown("### Robustness Test Scorecard")

    st.dataframe(
        formatted_scorecard,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(
        "Research limitations and score adjustments"
    ):

        st.dataframe(
            limitations_data,
            use_container_width=True,
            hide_index=True,
        )

    raw_score_text = str(
        assessment_data.loc[
            "Raw Robustness Score",
            "Result",
        ]
    )

    adjusted_score_text = str(
        assessment_data.loc[
            "Adjusted Research-Confidence Score",
            "Result",
        ]
    )

    confidence_text = str(
        assessment_data.loc[
            "Confidence Classification",
            "Result",
        ]
    )

    final_col_1, final_col_2, final_col_3 = (
        st.columns(3)
    )

    final_col_1.metric(
        "Raw Score",
        raw_score_text,
    )

    final_col_2.metric(
        "Adjusted Score",
        adjusted_score_text,
    )

    final_col_3.metric(
        "Research Confidence",
        confidence_text,
    )

    st.success(
        "The Adaptive Barbell strategy is classified as a "
        "risk-managed Indian equity allocation strategy with "
        "moderate–high research confidence."
    )

    st.info(
        "The confidence score is an internal research framework. "
        "It is not a probability of future success or a guarantee "
        "of future investment returns."
    )

# ---------------------------------------------------------
# v0.3.1 complete robustness and attribution
# ---------------------------------------------------------

st.markdown("---")
st.markdown("# Complete Robustness and Attribution")

st.caption(
    "This section completes the original v0.3 roadmap with "
    "bull/bear analysis, volatility-state analysis, performance "
    "attribution and the final four-category robustness classification."
)

(
    market_state_tab,
    attribution_tab,
    counterfactual_tab,
    final_classification_tab,
) = st.tabs(
    [
        "Market States",
        "Performance Attribution",
        "Counterfactual Analysis",
        "Final Classification",
    ]
)


# =========================================================
# TAB 1 — Bull/bear and volatility states
# =========================================================

with market_state_tab:

    bull_bear_data = robustness_data[
        "bull_bear_performance"
    ].copy()

    volatility_state_data = robustness_data[
        "volatility_state_performance"
    ].copy()

    adaptive_name_v031 = "Adaptive Barbell Strategy"
    fixed_name_v031 = "Fixed India 10 Portfolio"

    st.markdown("## Bull and Bear Market Analysis")

    bull_bear_display = bull_bear_data[
        [
            "Market State",
            "Investment",
            "Trading Days",
            "Annualised Return",
            "Annualised Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
        ]
    ].copy().astype(object)

    for row_number in bull_bear_display.index:

        bull_bear_display.loc[
            row_number,
            "Trading Days",
        ] = (
            f"{int(bull_bear_data.loc[row_number, 'Trading Days']):,}"
        )

        for metric in [
            "Annualised Return",
            "Annualised Volatility",
            "Maximum Drawdown",
        ]:

            bull_bear_display.loc[
                row_number,
                metric,
            ] = (
                f"{bull_bear_data.loc[row_number, metric]:.2%}"
            )

        bull_bear_display.loc[
            row_number,
            "Sharpe Ratio",
        ] = (
            f"{bull_bear_data.loc[row_number, 'Sharpe Ratio']:.2f}"
        )

    st.dataframe(
        bull_bear_display,
        use_container_width=True,
        hide_index=True,
    )

    bull_bear_indexed_v031 = (
        bull_bear_data
        .set_index(
            [
                "Market State",
                "Investment",
            ]
        )
    )

    bear_adaptive = bull_bear_indexed_v031.loc[
        (
            "Bear",
            adaptive_name_v031,
        )
    ]

    bear_fixed = bull_bear_indexed_v031.loc[
        (
            "Bear",
            fixed_name_v031,
        )
    ]

    bear_col_1, bear_col_2, bear_col_3, bear_col_4 = (
        st.columns(4)
    )

    bear_col_1.metric(
        "Bear-State Adaptive Return",
        f"{bear_adaptive['Annualised Return']:.2%}",
    )

    bear_col_2.metric(
        "Bear-State Fixed Return",
        f"{bear_fixed['Annualised Return']:.2%}",
    )

    bear_col_3.metric(
        "Adaptive Bear Volatility",
        f"{bear_adaptive['Annualised Volatility']:.2%}",
    )

    bear_col_4.metric(
        "Adaptive Bear Drawdown",
        f"{bear_adaptive['Maximum Drawdown']:.2%}",
    )

    st.info(
        "Bear states are defined using the previous trading day's "
        "trailing one-year Nifty 50 return. These are conditional "
        "statistics, not returns from a separately investable portfolio."
    )

    st.markdown("## High- and Low-Volatility Analysis")

    volatility_display = volatility_state_data[
        [
            "Volatility State",
            "Investment",
            "Trading Days",
            "Annualised Return",
            "Annualised Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
        ]
    ].copy().astype(object)

    for row_number in volatility_display.index:

        volatility_display.loc[
            row_number,
            "Trading Days",
        ] = (
            f"{int(volatility_state_data.loc[row_number, 'Trading Days']):,}"
        )

        for metric in [
            "Annualised Return",
            "Annualised Volatility",
            "Maximum Drawdown",
        ]:

            volatility_display.loc[
                row_number,
                metric,
            ] = (
                f"{volatility_state_data.loc[row_number, metric]:.2%}"
            )

        volatility_display.loc[
            row_number,
            "Sharpe Ratio",
        ] = (
            f"{volatility_state_data.loc[row_number, 'Sharpe Ratio']:.2f}"
        )

    st.dataframe(
        volatility_display,
        use_container_width=True,
        hide_index=True,
    )

    volatility_indexed_v031 = (
        volatility_state_data
        .set_index(
            [
                "Volatility State",
                "Investment",
            ]
        )
    )

    high_vol_adaptive = volatility_indexed_v031.loc[
        (
            "High Volatility",
            adaptive_name_v031,
        )
    ]

    high_vol_fixed = volatility_indexed_v031.loc[
        (
            "High Volatility",
            fixed_name_v031,
        )
    ]

    high_vol_col_1, high_vol_col_2, high_vol_col_3 = (
        st.columns(3)
    )

    high_vol_col_1.metric(
        "High-Vol Adaptive Sharpe",
        f"{high_vol_adaptive['Sharpe Ratio']:.2f}",
    )

    high_vol_col_2.metric(
        "High-Vol Fixed Sharpe",
        f"{high_vol_fixed['Sharpe Ratio']:.2f}",
    )

    high_vol_col_3.metric(
        "High-Vol Drawdown Improvement",
        (
            f"{high_vol_adaptive['Maximum Drawdown'] - high_vol_fixed['Maximum Drawdown']:+.2%}"
        ),
    )

    st.markdown(
        """
        The Adaptive strategy reduced volatility in bull, bear,
        high-volatility and low-volatility market states.

        Its strongest and most consistent advantage was therefore
        downside-risk control rather than guaranteed return
        outperformance in every market environment.
        """
    )


# =========================================================
# TAB 2 — Holding, bucket and sector attribution
# =========================================================

with attribution_tab:

    holding_data = robustness_data[
        "holding_attribution"
    ].copy()

    bucket_data = robustness_data[
        "bucket_attribution"
    ].copy()

    sector_data = robustness_data[
        "sector_attribution"
    ].copy()

    st.markdown("## Holding-Level Attribution")

    holding_display = (
        holding_data
        .copy()
        .astype(object)
    )

    for ticker in holding_display.index:

        for metric in [
            "Average Portfolio Weight",
            "Linked Return Contribution",
            "Contribution Share",
        ]:

            holding_display.loc[
                ticker,
                metric,
            ] = (
                f"{holding_data.loc[ticker, metric]:.2%}"
            )

    st.dataframe(
        holding_display,
        use_container_width=True,
    )

    top_two_holdings = (
        holding_data
        .sort_values(
            "Linked Return Contribution",
            ascending=False,
        )
        .head(2)
    )

    top_two_share = (
        top_two_holdings[
            "Contribution Share"
        ].sum()
    )

    top_holding_names = ", ".join(
        top_two_holdings.index.tolist()
    )

    holding_col_1, holding_col_2, holding_col_3 = (
        st.columns(3)
    )

    holding_col_1.metric(
        "Largest Contributor",
        top_two_holdings.index[0],
    )

    holding_col_2.metric(
        "Top-Two Contribution Share",
        f"{top_two_share:.2%}",
    )

    holding_col_3.metric(
        "Negative Contributors",
        (
            f"{(holding_data['Linked Return Contribution'] < 0).sum()}"
        ),
    )

    st.warning(
        f"The two largest contributors were {top_holding_names}. "
        f"Together they generated {top_two_share:.2%} of total "
        "linked return contribution, indicating material return "
        "concentration."
    )

    st.markdown("## Portfolio-Bucket Attribution")

    bucket_display = (
        bucket_data
        .copy()
        .astype(object)
    )

    for bucket_name in bucket_display.index:

        for metric in [
            "Linked Return Contribution",
            "Contribution Share",
        ]:

            bucket_display.loc[
                bucket_name,
                metric,
            ] = (
                f"{bucket_data.loc[bucket_name, metric]:.2%}"
            )

    st.dataframe(
        bucket_display,
        use_container_width=True,
    )

    growth_share = bucket_data.loc[
        "Growth Leaders",
        "Contribution Share",
    ]

    resilient_share = bucket_data.loc[
        "Resilient Compounders",
        "Contribution Share",
    ]

    bucket_col_1, bucket_col_2, bucket_col_3 = (
        st.columns(3)
    )

    bucket_col_1.metric(
        "Growth Leaders Share",
        f"{growth_share:.2%}",
    )

    bucket_col_2.metric(
        "Resilient Share",
        f"{resilient_share:.2%}",
    )

    bucket_col_3.metric(
        "Cash Contribution",
        (
            f"{bucket_data.loc['Cash Protection', 'Linked Return Contribution']:.2%}"
        ),
    )

    st.markdown("## Sector Attribution")

    sector_display = (
        sector_data
        .copy()
        .astype(object)
    )

    for sector_name in sector_display.index:

        for metric in [
            "Linked Return Contribution",
            "Contribution Share",
        ]:

            sector_display.loc[
                sector_name,
                metric,
            ] = (
                f"{sector_data.loc[sector_name, metric]:.2%}"
            )

    st.dataframe(
        sector_display,
        use_container_width=True,
    )

    st.markdown(
        """
        The attribution shows that most absolute return came from the
        Growth Leaders bucket and a limited number of high-performing
        holdings.

        The portfolio's risk reduction was broader than its return
        generation, which remained concentrated.
        """
    )


# =========================================================
# TAB 3 — Counterfactual portfolio analysis
# =========================================================

with counterfactual_tab:

    counterfactual_data = robustness_data[
        "counterfactual_performance"
    ].copy()

    incremental_data = robustness_data[
        "incremental_attribution"
    ].copy()

    st.markdown("## Counterfactual Portfolio Performance")

    counterfactual_metrics = [
        "Ending Value (₹)",
        "Total Return",
        "CAGR",
        "Annualised Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Calmar Ratio",
        "Beta vs Nifty 50",
    ]

    counterfactual_display_v031 = (
        counterfactual_data
        .loc[counterfactual_metrics]
        .copy()
        .astype(object)
    )

    for portfolio_name in counterfactual_display_v031.columns:

        counterfactual_display_v031.loc[
            "Ending Value (₹)",
            portfolio_name,
        ] = (
            f"₹{counterfactual_data.loc['Ending Value (₹)', portfolio_name]:,.0f}"
        )

        for metric in [
            "Total Return",
            "CAGR",
            "Annualised Volatility",
            "Maximum Drawdown",
        ]:

            counterfactual_display_v031.loc[
                metric,
                portfolio_name,
            ] = (
                f"{counterfactual_data.loc[metric, portfolio_name]:.2%}"
            )

        for metric in [
            "Sharpe Ratio",
            "Calmar Ratio",
            "Beta vs Nifty 50",
        ]:

            counterfactual_display_v031.loc[
                metric,
                portfolio_name,
            ] = (
                f"{counterfactual_data.loc[metric, portfolio_name]:.2f}"
            )

    st.dataframe(
        counterfactual_display_v031,
        use_container_width=True,
    )

    st.markdown("## Incremental Strategy Effects")

    incremental_display = (
        incremental_data
        .copy()
        .astype(object)
    )

    for effect_name in incremental_display.index:

        incremental_display.loc[
            effect_name,
            "Ending Value Effect (₹)",
        ] = (
            f"₹{incremental_data.loc[effect_name, 'Ending Value Effect (₹)']:+,.0f}"
        )

        for metric in [
            "CAGR Effect",
            "Volatility Effect",
            "Drawdown Improvement",
        ]:

            incremental_display.loc[
                effect_name,
                metric,
            ] = (
                f"{incremental_data.loc[effect_name, metric]:+.2%}"
            )

        for metric in [
            "Sharpe Effect",
            "Calmar Effect",
            "Beta Effect",
        ]:

            incremental_display.loc[
                effect_name,
                metric,
            ] = (
                f"{incremental_data.loc[effect_name, metric]:+.2f}"
            )

    st.dataframe(
        incremental_display,
        use_container_width=True,
    )

    st.markdown(
        """
        The selected ten-stock universe generated most of the absolute
        return advantage over the Nifty 50.

        Dynamic weighting and defensive cash protection added value
        mainly through lower volatility, smaller drawdowns, lower beta
        and stronger risk-adjusted performance.
        """
    )


# =========================================================
# TAB 4 — Final roadmap-aligned classification
# =========================================================

with final_classification_tab:

    classification_data = robustness_data[
        "classification_evidence"
    ].copy()

    classification_summary = robustness_data[
        "final_robustness_classification"
    ].copy()

    final_classification = str(
        classification_summary.loc[
            "Final Robustness Classification",
            "Result",
        ]
    )

    combinations_tested = int(
        float(
            classification_summary.loc[
                "Parameter Combinations Tested",
                "Result",
            ]
        )
    )

    top_two_concentration = float(
        classification_summary.loc[
            "Top-Two Holding Contribution Share",
            "Result",
        ]
    )

    growth_concentration = float(
        classification_summary.loc[
            "Growth Bucket Contribution Share",
            "Result",
        ]
    )

    st.markdown("## Final Robustness Classification")

    final_class_col_1, final_class_col_2, final_class_col_3 = (
        st.columns(3)
    )

    final_class_col_1.metric(
        "Classification",
        final_classification,
    )

    final_class_col_2.metric(
        "Parameter Combinations",
        f"{combinations_tested:,}",
    )

    final_class_col_3.metric(
        "Top-Two Contribution Share",
        f"{top_two_concentration:.2%}",
    )

    classification_display = (
        classification_data
        .copy()
    )

    classification_display[
        "Result"
    ] = classification_display[
        "Result"
    ].map(
        lambda value: (
            "Passed"
            if str(value).strip().lower() == "true"
            else "Not passed"
        )
    )

    st.dataframe(
        classification_display,
        use_container_width=True,
        hide_index=True,
    )

    if final_classification == "Robust":

        st.success(
            "The strategy is classified as Robust."
        )

    elif final_classification == "Moderately robust":

        st.warning(
            "The strategy is classified as Moderately robust."
        )

    elif final_classification == "Parameter-sensitive":

        st.error(
            "The strategy is classified as Parameter-sensitive."
        )

    else:

        st.error(
            "The strategy is classified as Likely overfitted."
        )

    st.markdown(
        f"""
        The strategy passed the completed **{combinations_tested}**
        combination parameter grid and remained resilient under higher
        transaction costs.

        It also reduced risk across different market environments.

        However, the Growth Leaders bucket generated
        **{growth_concentration:.2%}** of total linked return
        contribution, and the two largest holdings generated
        **{top_two_concentration:.2%}**.

        Point-in-time universe construction and external-universe
        validation are still missing.

        The academically defensible conclusion is therefore:

        **Moderately robust — strong risk-control evidence, but
        meaningful stock-selection bias and return-concentration
        limitations remain.**
        """
    )

    st.info(
        "This classification applies to the tested India 10 research "
        "case. It is not a guarantee of future performance and is not "
        "a probability estimate."
    )
