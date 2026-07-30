
from pathlib import Path

import pandas as pd
import streamlit as st


REPOSITORY_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

FORECASTING_DATA_DIRECTORY = (
    REPOSITORY_ROOT
    / "data"
    / "processed"
    / "forecasting"
)


FORECASTING_FILE_MAP = {
    "stationarity_results":
        "stationarity_results.csv",

    "arima_order_selection":
        "arima_order_selection.csv",

    "arima_residual_diagnostics":
        "arima_residual_diagnostics.csv",

    "arima_forecast_accuracy":
        "arima_forecast_accuracy.csv",

    "arima_interval_summary":
        "arima_interval_summary.csv",

    "volatility_model_selection":
        "volatility_model_selection.csv",

    "garch_residual_diagnostics":
        "garch_residual_diagnostics.csv",

    "volatility_forecast_accuracy":
        "volatility_forecast_accuracy.csv",

    "latest_stock_volatility_forecasts":
        "latest_stock_volatility_forecasts.csv",

    "latest_forecast_regime_summary":
        "latest_forecast_regime_summary.csv",

    "latest_forecast_aware_allocation":
        "latest_forecast_aware_allocation.csv",

    "monthly_stock_volatility_forecasts":
        "monthly_stock_volatility_forecasts.csv",

    "monthly_forecast_aware_targets":
        "monthly_forecast_aware_targets.csv",

    "forecast_aware_backtest_daily":
        "forecast_aware_backtest_daily.csv",

    "historical_volatility_backtest_daily":
        "historical_volatility_backtest_daily.csv",

    "volatility_allocation_comparison":
        "volatility_allocation_comparison.csv",

    "volatility_model_trading_comparison":
        "volatility_model_trading_comparison.csv",

    "forecast_aware_incremental_effect":
        "forecast_aware_incremental_effect.csv",

    "final_volatility_model_roles":
        "final_volatility_model_roles.csv",

    "v04_research_summary":
        "v04_research_summary.csv",
}


INDEX_COLUMN_DATASETS = {
    "stationarity_results": 0,
    "arima_residual_diagnostics": 0,
    "arima_forecast_accuracy": 0,
    "arima_interval_summary": 0,
    "volatility_model_selection": 0,
    "garch_residual_diagnostics": 0,
    "volatility_forecast_accuracy": 0,
    "latest_stock_volatility_forecasts": 0,
    "latest_forecast_aware_allocation": 0,
    "volatility_allocation_comparison": 0,
    "volatility_model_trading_comparison": 0,
    "forecast_aware_incremental_effect": 0,
}


DATE_COLUMNS_BY_DATASET = {
    "latest_forecast_regime_summary": [
        "Signal Date",
    ],

    "monthly_stock_volatility_forecasts": [
        "Signal Date",
        "Execution Date",
    ],

    "monthly_forecast_aware_targets": [
        "Execution Date",
        "Signal Date",
        "Model Data Cutoff",
    ],

    "forecast_aware_backtest_daily": [
        "Date",
    ],

    "historical_volatility_backtest_daily": [
        "Date",
    ],
}


def validate_forecasting_files():
    """
    Confirm that every required v0.4 forecasting CSV exists.
    """

    missing_files = []

    for filename in FORECASTING_FILE_MAP.values():

        file_path = (
            FORECASTING_DATA_DIRECTORY
            / filename
        )

        if not file_path.exists():
            missing_files.append(
                str(file_path)
            )

    if missing_files:
        raise FileNotFoundError(
            "Missing v0.4 forecasting files:\n"
            + "\n".join(missing_files)
        )

    return True


@st.cache_data
def load_forecasting_data():
    """
    Load all precomputed v0.4 forecasting and risk datasets.
    """

    validate_forecasting_files()

    forecasting_data = {}

    for dataset_name, filename in (
        FORECASTING_FILE_MAP.items()
    ):

        file_path = (
            FORECASTING_DATA_DIRECTORY
            / filename
        )

        read_options = {}

        if dataset_name in INDEX_COLUMN_DATASETS:

            read_options["index_col"] = (
                INDEX_COLUMN_DATASETS[
                    dataset_name
                ]
            )

        dataset = pd.read_csv(
            file_path,
            **read_options,
        )

        for date_column in (
            DATE_COLUMNS_BY_DATASET.get(
                dataset_name,
                [],
            )
        ):

            if date_column in dataset.columns:

                dataset[date_column] = (
                    pd.to_datetime(
                        dataset[date_column],
                        errors="coerce",
                    )
                )

        forecasting_data[
            dataset_name
        ] = dataset

    return forecasting_data


def build_forecasting_manifest(
    forecasting_data,
):
    """
    Build a compact dataset manifest for diagnostics.
    """

    manifest_records = []

    for dataset_name, dataset in (
        forecasting_data.items()
    ):

        manifest_records.append(
            {
                "Dataset": dataset_name,
                "Rows": len(dataset),
                "Columns": len(
                    dataset.columns
                ),
            }
        )

    return (
        pd.DataFrame(
            manifest_records
        )
        .sort_values("Dataset")
        .reset_index(drop=True)
    )
