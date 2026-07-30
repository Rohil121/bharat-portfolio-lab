
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Bharat Portfolio Lab",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------
# App heading
# ---------------------------------------------------------

st.title("📊 Bharat Portfolio Lab")

st.subheader("India 10 Adaptive Barbell Portfolio")

st.markdown(
    """
    An explainable Indian equity portfolio analytics platform combining:

    - five **Resilient Compounders**,
    - five **Growth Leaders**,
    - portfolio risk and performance analysis, and
    - comparison against the Nifty 50.

    **Version v0.1:** Preliminary deployed model.
    """
)

st.info(
    "This application is an academic financial-analytics project "
    "and does not constitute investment advice."
)


# ---------------------------------------------------------
# Portfolio definition
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

portfolio["Weight"] = 1 / len(portfolio)

portfolio_tickers = portfolio["Ticker"].tolist()

BENCHMARK_TICKER = "^NSEI"
BENCHMARK_NAME = "Nifty 50"

TRADING_DAYS = 252
RISK_FREE_RATE = 0.065


# ---------------------------------------------------------
# User inputs
# ---------------------------------------------------------

st.sidebar.header("Model Settings")

start_date = st.sidebar.date_input(
    "Analysis start date",
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
    ### Current model

    - Equal initial weights
    - Buy-and-hold portfolio
    - Nifty 50 benchmark
    - 6.5% provisional risk-free rate
    """
)


# ---------------------------------------------------------
# Data download
# ---------------------------------------------------------

@st.cache_data(ttl=3600)
def download_market_data(tickers, benchmark, start):
    all_tickers = tickers + [benchmark]

    data = yf.download(
        tickers=all_tickers,
        start=str(start),
        end=(
            pd.Timestamp.today().normalize()
            + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = all_tickers

    prices = prices.reindex(columns=all_tickers)
    prices = prices.sort_index()
    prices = prices.ffill(limit=3).dropna()

    return prices


try:
    with st.spinner("Downloading Indian market data..."):
        close_prices = download_market_data(
            portfolio_tickers,
            BENCHMARK_TICKER,
            start_date,
        )

except Exception as error:
    st.error(f"Market data could not be downloaded: {error}")
    st.stop()


if close_prices.empty:
    st.error("No market data is available for the selected period.")
    st.stop()


# ---------------------------------------------------------
# Portfolio performance
# ---------------------------------------------------------

stock_prices = close_prices[portfolio_tickers]

benchmark_price = close_prices[BENCHMARK_TICKER]

weights = portfolio.set_index("Ticker")["Weight"]

initial_stock_prices = stock_prices.iloc[0]
initial_allocations = weights * initial_capital
shares_purchased = initial_allocations / initial_stock_prices

holding_values = stock_prices.mul(shares_purchased, axis=1)

portfolio_value = holding_values.sum(axis=1)
portfolio_value.name = "India 10 Portfolio"

benchmark_units = initial_capital / benchmark_price.iloc[0]
benchmark_value = benchmark_price * benchmark_units
benchmark_value.name = BENCHMARK_NAME

performance_values = pd.concat(
    [portfolio_value, benchmark_value],
    axis=1,
).dropna()

daily_returns = performance_values.pct_change().dropna()


# ---------------------------------------------------------
# Metric functions
# ---------------------------------------------------------

def calculate_metrics(value_series, return_series):
    years = (
        value_series.index[-1] - value_series.index[0]
    ).days / 365.25

    total_return = (
        value_series.iloc[-1] / value_series.iloc[0] - 1
    )

    cagr = (
        value_series.iloc[-1] / value_series.iloc[0]
    ) ** (1 / years) - 1

    annualised_volatility = (
        return_series.std() * np.sqrt(TRADING_DAYS)
    )

    daily_rf = (
        (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    )

    annualised_excess_return = (
        (return_series - daily_rf).mean() * TRADING_DAYS
    )

    sharpe_ratio = (
        annualised_excess_return / annualised_volatility
        if annualised_volatility > 0
        else np.nan
    )

    drawdown = value_series / value_series.cummax() - 1
    maximum_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": annualised_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": maximum_drawdown,
    }


portfolio_metrics = calculate_metrics(
    performance_values["India 10 Portfolio"],
    daily_returns["India 10 Portfolio"],
)

benchmark_metrics = calculate_metrics(
    performance_values[BENCHMARK_NAME],
    daily_returns[BENCHMARK_NAME],
)


# ---------------------------------------------------------
# Dashboard metrics
# ---------------------------------------------------------

st.markdown("## Portfolio Overview")

column_1, column_2, column_3, column_4 = st.columns(4)

column_1.metric(
    "Current Portfolio Value",
    f"₹{portfolio_value.iloc[-1]:,.0f}",
)

column_2.metric(
    "Portfolio CAGR",
    f"{portfolio_metrics['CAGR']:.2%}",
)

column_3.metric(
    "Annualised Volatility",
    f"{portfolio_metrics['Volatility']:.2%}",
)

column_4.metric(
    "Sharpe Ratio",
    f"{portfolio_metrics['Sharpe Ratio']:.2f}",
)


# ---------------------------------------------------------
# Portfolio performance chart
# ---------------------------------------------------------

st.markdown("## Portfolio vs Nifty 50")

figure, axis = plt.subplots(figsize=(12, 6))

axis.plot(
    performance_values.index,
    performance_values["India 10 Portfolio"],
    linewidth=2,
    label="India 10 Portfolio",
)

axis.plot(
    performance_values.index,
    performance_values[BENCHMARK_NAME],
    linewidth=2,
    linestyle="--",
    label=BENCHMARK_NAME,
)

axis.set_title(
    f"Growth of ₹{initial_capital:,.0f}"
)

axis.set_xlabel("Date")
axis.set_ylabel("Investment Value (₹)")
axis.legend()
axis.grid(alpha=0.25)

st.pyplot(figure)


# ---------------------------------------------------------
# Risk comparison table
# ---------------------------------------------------------

st.markdown("## Risk and Performance Comparison")

comparison_table = pd.DataFrame(
    {
        "India 10 Portfolio": portfolio_metrics,
        BENCHMARK_NAME: benchmark_metrics,
    }
)

# Create a separate object-type table for formatted text
# Pandas 3.x does not allow strings inside float64 columns.
display_table = comparison_table.copy().astype(object)

percentage_rows = [
    "Total Return",
    "CAGR",
    "Volatility",
    "Maximum Drawdown",
]

for row in percentage_rows:
    display_table.loc[row, :] = (
        comparison_table.loc[row]
        .astype(float)
        .map(lambda value: f"{value:.2%}")
    )

display_table.loc["Sharpe Ratio", :] = (
    comparison_table.loc["Sharpe Ratio"]
    .astype(float)
    .map(lambda value: f"{value:.2f}")
)

st.dataframe(
    display_table,
    use_container_width=True,
)


# ---------------------------------------------------------
# Portfolio composition
# ---------------------------------------------------------

st.markdown("## Flagship Portfolio Holdings")

portfolio_display = portfolio[
    [
        "Company",
        "Ticker",
        "Sector",
        "Strategy Bucket",
        "Weight",
    ]
].copy()

portfolio_display["Weight"] = (
    portfolio_display["Weight"]
    .map(lambda value: f"{value:.2%}")
)

st.dataframe(
    portfolio_display,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# Drawdown chart
# ---------------------------------------------------------

st.markdown("## Historical Drawdown")

drawdowns = (
    performance_values
    .div(performance_values.cummax())
    .sub(1)
)

figure_drawdown, axis_drawdown = plt.subplots(
    figsize=(12, 5)
)

axis_drawdown.plot(
    drawdowns.index,
    drawdowns["India 10 Portfolio"],
    linewidth=2,
    label="India 10 Portfolio",
)

axis_drawdown.plot(
    drawdowns.index,
    drawdowns[BENCHMARK_NAME],
    linewidth=2,
    linestyle="--",
    label=BENCHMARK_NAME,
)

axis_drawdown.set_xlabel("Date")
axis_drawdown.set_ylabel("Drawdown")
axis_drawdown.legend()
axis_drawdown.grid(alpha=0.25)

st.pyplot(figure_drawdown)


# ---------------------------------------------------------
# Current limitations
# ---------------------------------------------------------

st.markdown("## Current Model Limitations")

st.markdown(
    """
    - The present portfolio is a fixed case-study portfolio.
    - Historical constituents are not yet reconstructed.
    - Taxes and complete Indian transaction costs are not yet modelled.
    - The deployed version currently presents the buy-and-hold portfolio.
    - The Adaptive Barbell backtest will be integrated into the next release.
    """
)
