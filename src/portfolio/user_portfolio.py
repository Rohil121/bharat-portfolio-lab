"""
User-selected Indian equity portfolio input and market-data engine.

Supported exchanges:
    NSE through the .NS suffix
    BSE through the .BO suffix

Supported input modes:
    Weight
    Invested Amount
    Quantity
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yfinance as yf


SUPPORTED_EXCHANGES = {
    "NSE": ".NS",
    "BSE": ".BO",
}

SUPPORTED_INPUT_MODES = {
    "WEIGHT": "Weight",
    "INVESTED AMOUNT": "Invested Amount",
    "QUANTITY": "Quantity",
}



def standardise_indian_ticker(
    ticker: str,
    exchange: str = "NSE",
) -> str:
    """
    Convert a user-entered Indian trading symbol into
    Yahoo Finance format.

    NSE example:
        HDFCBANK becomes HDFCBANK.NS

    BSE example:
        HDFCBANK becomes HDFCBANK.BO

    Numeric BSE scrip codes are rejected because the current
    market-data source expects the BSE trading symbol.
    """

    clean_ticker = str(
        ticker
    ).strip().upper()

    clean_exchange = str(
        exchange
    ).strip().upper()

    if not clean_ticker:
        raise ValueError(
            "Ticker cannot be blank."
        )

    has_nse_suffix = (
        clean_ticker.endswith(
            ".NS"
        )
    )

    has_bse_suffix = (
        clean_ticker.endswith(
            ".BO"
        )
    )

    if has_nse_suffix or has_bse_suffix:

        ticker_without_suffix = (
            clean_ticker[:-3]
        )

        if (
            has_bse_suffix
            and ticker_without_suffix.isdigit()
        ):
            raise ValueError(
                "Numeric BSE scrip codes are not supported by "
                "the current market-data source. Enter the BSE "
                "trading symbol instead. For example, enter "
                "HDFCBANK rather than 500180."
            )

        return clean_ticker

    if clean_exchange not in SUPPORTED_EXCHANGES:
        raise ValueError(
            f"Unsupported exchange: {exchange}. "
            "Use NSE or BSE."
        )

    if (
        clean_exchange == "BSE"
        and clean_ticker.isdigit()
    ):
        raise ValueError(
            "Numeric BSE scrip codes are not supported by "
            "the current market-data source. Enter the BSE "
            "trading symbol instead. For example, enter "
            "HDFCBANK rather than 500180."
        )

    return (
        clean_ticker
        + SUPPORTED_EXCHANGES[
            clean_exchange
        ]
    )


def validate_analysis_dates(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Validate and standardise the selected analysis period.
    """

    start_timestamp = pd.Timestamp(
        start_date
    ).normalize()

    end_timestamp = pd.Timestamp(
        end_date
    ).normalize()

    if pd.isna(start_timestamp):
        raise ValueError(
            "The start date is invalid."
        )

    if pd.isna(end_timestamp):
        raise ValueError(
            "The end date is invalid."
        )

    if start_timestamp >= end_timestamp:
        raise ValueError(
            "The start date must be earlier than the end date."
        )

    return (
        start_timestamp,
        end_timestamp,
    )


def _extract_close_prices(
    raw_data: pd.DataFrame,
    requested_tickers: list[str],
) -> pd.DataFrame:
    """
    Extract adjusted closing prices from yfinance output.
    """

    if raw_data.empty:
        raise RuntimeError(
            "No market data were downloaded."
        )

    if isinstance(
        raw_data.columns,
        pd.MultiIndex,
    ):

        first_level = (
            raw_data.columns
            .get_level_values(0)
        )

        second_level = (
            raw_data.columns
            .get_level_values(1)
        )

        if "Close" in first_level:
            close_prices = (
                raw_data[
                    "Close"
                ]
                .copy()
            )

        elif "Close" in second_level:
            close_prices = (
                raw_data.xs(
                    "Close",
                    axis=1,
                    level=1,
                )
                .copy()
            )

        else:
            raise RuntimeError(
                "Closing-price data were not returned."
            )

    else:

        if "Close" not in raw_data.columns:
            raise RuntimeError(
                "Closing-price data were not returned."
            )

        if len(requested_tickers) != 1:
            raise RuntimeError(
                "Unexpected single-level market-data structure."
            )

        close_prices = pd.DataFrame(
            {
                requested_tickers[0]:
                    raw_data[
                        "Close"
                    ]
            }
        )

    close_prices.columns = [
        str(column).upper()
        for column in close_prices.columns
    ]

    return (
        close_prices
        .sort_index()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


def download_latest_stock_prices(
    tickers: list[str],
    lookback_period: str = "1mo",
) -> pd.DataFrame:
    """
    Download the latest available adjusted close for each stock.
    """

    unique_tickers = list(
        dict.fromkeys(
            str(ticker).upper()
            for ticker in tickers
        )
    )

    if not unique_tickers:
        raise ValueError(
            "At least one ticker is required."
        )

    raw_data = yf.download(
        tickers=unique_tickers,
        period=lookback_period,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    close_prices = _extract_close_prices(
        raw_data=raw_data,
        requested_tickers=unique_tickers,
    )

    latest_price_records = []

    for ticker in unique_tickers:

        if ticker not in close_prices.columns:
            raise ValueError(
                f"No recent price was found for {ticker}."
            )

        ticker_prices = (
            close_prices[
                ticker
            ]
            .dropna()
        )

        if ticker_prices.empty:
            raise ValueError(
                f"No valid recent price was found for {ticker}."
            )

        latest_price_records.append(
            {
                "Ticker":
                    ticker,

                "Latest Price (₹)":
                    float(
                        ticker_prices.iloc[-1]
                    ),

                "Price Date":
                    pd.Timestamp(
                        ticker_prices.index[-1]
                    ),
            }
        )

    return (
        pd.DataFrame(
            latest_price_records
        )
        .set_index("Ticker")
    )


def construct_user_portfolio(
    portfolio_input: pd.DataFrame,
    input_mode: str,
) -> dict[str, Any]:
    """
    Convert user-entered holdings into normalised weights.

    Required columns
    ----------------
    Ticker
    Value

    Optional column
    ---------------
    Exchange

    The Value column represents:
        percentage/relative weight in Weight mode,
        Indian rupees in Invested Amount mode,
        shares in Quantity mode.
    """

    if not isinstance(
        portfolio_input,
        pd.DataFrame,
    ):
        raise TypeError(
            "Portfolio input must be a pandas DataFrame."
        )

    required_columns = {
        "Ticker",
        "Value",
    }

    missing_columns = (
        required_columns
        - set(
            portfolio_input.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    portfolio_table = (
        portfolio_input
        .copy()
        .dropna(
            how="all"
        )
    )

    if "Exchange" not in portfolio_table.columns:
        portfolio_table[
            "Exchange"
        ] = "NSE"

    portfolio_table[
        "Standardised Ticker"
    ] = [
        standardise_indian_ticker(
            ticker=ticker,
            exchange=exchange,
        )
        for ticker, exchange in zip(
            portfolio_table[
                "Ticker"
            ],
            portfolio_table[
                "Exchange"
            ],
        )
    ]

    duplicated_tickers = (
        portfolio_table.loc[
            portfolio_table[
                "Standardised Ticker"
            ].duplicated(
                keep=False
            ),
            "Standardised Ticker",
        ]
        .unique()
        .tolist()
    )

    if duplicated_tickers:
        raise ValueError(
            "Duplicate holdings found: "
            + ", ".join(
                duplicated_tickers
            )
        )

    portfolio_table[
        "Value"
    ] = pd.to_numeric(
        portfolio_table[
            "Value"
        ],
        errors="coerce",
    )

    if portfolio_table[
        "Value"
    ].isna().any():
        raise ValueError(
            "All portfolio values must be numeric."
        )

    if (
        portfolio_table[
            "Value"
        ]
        <= 0
    ).any():
        raise ValueError(
            "All portfolio values must be greater than zero."
        )

    if len(portfolio_table) < 2:
        raise ValueError(
            "Select at least two listed stocks."
        )

    normalised_mode = str(
        input_mode
    ).strip().upper()

    if normalised_mode not in SUPPORTED_INPUT_MODES:
        raise ValueError(
            "Input mode must be Weight, "
            "Invested Amount or Quantity."
        )

    selected_mode = (
        SUPPORTED_INPUT_MODES[
            normalised_mode
        ]
    )

    latest_prices = None

    if selected_mode == "Weight":

        portfolio_table[
            "Market Value (₹)"
        ] = np.nan

        portfolio_table[
            "Calculated Weight"
        ] = (
            portfolio_table[
                "Value"
            ]
            / portfolio_table[
                "Value"
            ].sum()
        )

    elif selected_mode == "Invested Amount":

        portfolio_table[
            "Market Value (₹)"
        ] = portfolio_table[
            "Value"
        ]

        portfolio_table[
            "Calculated Weight"
        ] = (
            portfolio_table[
                "Market Value (₹)"
            ]
            / portfolio_table[
                "Market Value (₹)"
            ].sum()
        )

    else:

        latest_prices = (
            download_latest_stock_prices(
                portfolio_table[
                    "Standardised Ticker"
                ].tolist()
            )
        )

        portfolio_table = (
            portfolio_table.merge(
                latest_prices.reset_index(),
                left_on="Standardised Ticker",
                right_on="Ticker",
                how="left",
                suffixes=(
                    "",
                    "_Price",
                ),
            )
        )

        portfolio_table.drop(
            columns=[
                "Ticker_Price",
            ],
            errors="ignore",
            inplace=True,
        )

        portfolio_table[
            "Quantity"
        ] = portfolio_table[
            "Value"
        ]

        portfolio_table[
            "Market Value (₹)"
        ] = (
            portfolio_table[
                "Quantity"
            ]
            * portfolio_table[
                "Latest Price (₹)"
            ]
        )

        portfolio_table[
            "Calculated Weight"
        ] = (
            portfolio_table[
                "Market Value (₹)"
            ]
            / portfolio_table[
                "Market Value (₹)"
            ].sum()
        )

    portfolio_weights = (
        portfolio_table
        .set_index(
            "Standardised Ticker"
        )[
            "Calculated Weight"
        ]
        .rename(
            "Portfolio Weight"
        )
    )

    total_market_value = (
        portfolio_table[
            "Market Value (₹)"
        ].sum()
        if portfolio_table[
            "Market Value (₹)"
        ].notna().any()
        else np.nan
    )

    portfolio_summary = pd.Series(
        {
            "Input Mode":
                selected_mode,

            "Number of Holdings":
                len(
                    portfolio_weights
                ),

            "Total Weight":
                portfolio_weights.sum(),

            "Largest Holding":
                portfolio_weights.max(),

            "Top-3 Concentration":
                portfolio_weights.nlargest(
                    3
                ).sum(),

            "Herfindahl Index":
                np.square(
                    portfolio_weights
                ).sum(),

            "Effective Holdings":
                (
                    1.0
                    / np.square(
                        portfolio_weights
                    ).sum()
                ),

            "Total Market Value (₹)":
                total_market_value,
        },
        name="User Portfolio",
    )

    if not np.isclose(
        portfolio_weights.sum(),
        1.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Calculated portfolio weights do not sum to 100%."
        )

    return {
        "weights":
            portfolio_weights,

        "portfolio_table":
            portfolio_table,

        "summary":
            portfolio_summary,

        "latest_prices":
            latest_prices,
    }


def download_user_market_data(
    portfolio_weights: Mapping[str, float] | pd.Series,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    benchmark: str = "^NSEI",
    minimum_observations: int = 252,
    forward_fill_limit: int = 3,
    allow_bse_history_fallback: bool = True,
) -> dict[str, Any]:
    """
    Download and align historical portfolio and benchmark data.

    The supplied end date is inclusive.

    For a selected BSE stock with insufficient historical data,
    the function may use the same trading symbol's NSE history.
    The portfolio continues to display the selected BSE ticker,
    while the substitution is disclosed in the data-quality table.
    """

    start_timestamp, end_timestamp = (
        validate_analysis_dates(
            start_date=start_date,
            end_date=end_date,
        )
    )

    weights = pd.Series(
        portfolio_weights,
        dtype=float,
        name="Portfolio Weight",
    )

    if len(weights) < 2:
        raise ValueError(
            "At least two portfolio holdings are required."
        )

    if weights.index.duplicated().any():
        raise ValueError(
            "Portfolio tickers must be unique."
        )

    if not np.isfinite(
        weights.to_numpy()
    ).all():
        raise ValueError(
            "Portfolio weights must be finite numbers."
        )

    if (
        weights
        <= 0
    ).any():
        raise ValueError(
            "Portfolio weights must be greater than zero."
        )

    weights.index = [
        str(ticker).upper()
        for ticker in weights.index
    ]

    weights = (
        weights
        / weights.sum()
    )

    benchmark_ticker = str(
        benchmark
    ).strip().upper()

    selected_tickers = (
        weights.index.tolist()
    )

    requested_tickers = (
        selected_tickers
        + [
            benchmark_ticker
        ]
    )

    inclusive_download_end = (
        end_timestamp
        + pd.Timedelta(
            days=1
        )
    )

    raw_data = yf.download(
        tickers=requested_tickers,
        start=start_timestamp.strftime(
            "%Y-%m-%d"
        ),
        end=inclusive_download_end.strftime(
            "%Y-%m-%d"
        ),
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    raw_close_prices = _extract_close_prices(
        raw_data=raw_data,
        requested_tickers=requested_tickers,
    )

    # Reindex creates blank columns for tickers that Yahoo omitted,
    # allowing them to be evaluated for fallback.
    raw_close_prices = (
        raw_close_prices
        .reindex(
            columns=requested_tickers
        )
        .loc[
            start_timestamp:
            end_timestamp
        ]
    )

    original_observations = {
        ticker: int(
            raw_close_prices[
                ticker
            ]
            .notna()
            .sum()
        )
        for ticker in requested_tickers
    }

    historical_source_map = {
        ticker: ticker
        for ticker in requested_tickers
    }

    fallback_used_map = {
        ticker: False
        for ticker in requested_tickers
    }

    fallback_pairs = {}

    if allow_bse_history_fallback:

        for selected_ticker in selected_tickers:

            if (
                selected_ticker.endswith(
                    ".BO"
                )
                and original_observations[
                    selected_ticker
                ]
                < minimum_observations
            ):

                fallback_pairs[
                    selected_ticker
                ] = (
                    selected_ticker[:-3]
                    + ".NS"
                )

    if fallback_pairs:

        fallback_tickers = list(
            dict.fromkeys(
                fallback_pairs.values()
            )
        )

        fallback_raw_data = yf.download(
            tickers=fallback_tickers,
            start=start_timestamp.strftime(
                "%Y-%m-%d"
            ),
            end=inclusive_download_end.strftime(
                "%Y-%m-%d"
            ),
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )

        if not fallback_raw_data.empty:

            fallback_prices = (
                _extract_close_prices(
                    raw_data=fallback_raw_data,
                    requested_tickers=fallback_tickers,
                )
                .reindex(
                    columns=fallback_tickers
                )
                .loc[
                    start_timestamp:
                    end_timestamp
                ]
            )

            for (
                selected_bse_ticker,
                fallback_nse_ticker,
            ) in fallback_pairs.items():

                if (
                    fallback_nse_ticker
                    not in fallback_prices.columns
                ):
                    continue

                fallback_series = (
                    fallback_prices[
                        fallback_nse_ticker
                    ]
                    .dropna()
                )

                if (
                    len(
                        fallback_series
                    )
                    >= minimum_observations
                ):

                    raw_close_prices[
                        selected_bse_ticker
                    ] = (
                        fallback_prices[
                            fallback_nse_ticker
                        ]
                        .reindex(
                            raw_close_prices.index
                        )
                    )

                    historical_source_map[
                        selected_bse_ticker
                    ] = fallback_nse_ticker

                    fallback_used_map[
                        selected_bse_ticker
                    ] = True

    quality_records = []

    total_rows = len(
        raw_close_prices
    )

    for ticker in requested_tickers:

        valid_observations = int(
            raw_close_prices[
                ticker
            ]
            .notna()
            .sum()
        )

        missing_observations = (
            total_rows
            - valid_observations
        )

        quality_records.append(
            {
                "Ticker":
                    ticker,

                "Original Valid Observations":
                    original_observations[
                        ticker
                    ],

                "Valid Observations":
                    valid_observations,

                "Missing Observations":
                    missing_observations,

                "Missing Percentage":
                    (
                        missing_observations
                        / total_rows
                        if total_rows > 0
                        else np.nan
                    ),

                "Sufficient History":
                    (
                        valid_observations
                        >= minimum_observations
                    ),

                "Historical Source":
                    historical_source_map[
                        ticker
                    ],

                "Fallback Used":
                    fallback_used_map[
                        ticker
                    ],
            }
        )

    data_quality = (
        pd.DataFrame(
            quality_records
        )
        .set_index("Ticker")
    )

    insufficient_tickers = (
        data_quality.loc[
            ~data_quality[
                "Sufficient History"
            ]
        ]
        .index
        .tolist()
    )

    if insufficient_tickers:
        raise ValueError(
            "Insufficient historical observations for: "
            + ", ".join(
                insufficient_tickers
            )
            + ". Try a later start date or an NSE-listed "
            "equivalent."
        )

    cleaned_prices = (
        raw_close_prices
        .ffill(
            limit=forward_fill_limit
        )
        .dropna()
    )

    if len(cleaned_prices) < minimum_observations:
        raise ValueError(
            "Insufficient common aligned history after "
            "removing missing observations."
        )

    portfolio_prices = (
        cleaned_prices[
            selected_tickers
        ]
        .copy()
    )

    benchmark_prices = (
        cleaned_prices[
            benchmark_ticker
        ]
        .rename(
            "Benchmark"
        )
    )

    portfolio_returns = (
        portfolio_prices
        .pct_change(
            fill_method=None
        )
        .dropna()
    )

    benchmark_returns = (
        benchmark_prices
        .pct_change(
            fill_method=None
        )
        .dropna()
    )

    aligned_returns = pd.concat(
        [
            portfolio_returns,
            benchmark_returns,
        ],
        axis=1,
    ).dropna()

    portfolio_returns = (
        aligned_returns[
            selected_tickers
        ]
    )

    benchmark_returns = (
        aligned_returns[
            "Benchmark"
        ]
    )

    if len(portfolio_returns) < (
        minimum_observations - 1
    ):
        raise ValueError(
            "Insufficient aligned return observations."
        )

    return {
        "weights":
            weights,

        "prices":
            cleaned_prices,

        "portfolio_prices":
            portfolio_prices,

        "benchmark_prices":
            benchmark_prices,

        "portfolio_returns":
            portfolio_returns,

        "benchmark_returns":
            benchmark_returns,

        "data_quality":
            data_quality,

        "historical_source_map":
            historical_source_map,

        "bse_history_fallback_used":
            bool(
                any(
                    fallback_used_map.values()
                )
            ),

        "benchmark_ticker":
            benchmark_ticker,

        "start_date":
            portfolio_returns.index.min(),

        "end_date":
            portfolio_returns.index.max(),
    }
