
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------

st.title("📊 Bharat Portfolio Lab")

st.subheader("India 10 Adaptive Barbell Portfolio")

st.markdown(
    """
    **Version v0.2 — Strategy Dashboard**

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
    "These results are preliminary. The stock universe was selected "
    "using present-day knowledge and has not yet undergone complete "
    "out-of-sample, survivorship-bias or robustness testing."
)
