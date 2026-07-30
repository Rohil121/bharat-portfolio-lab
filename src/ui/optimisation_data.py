from pathlib import Path

import pandas as pd
import streamlit as st


REPOSITORY_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

OPTIMISATION_DATA_DIRECTORY = (
    REPOSITORY_ROOT
    / "data"
    / "processed"
    / "optimisation"
)

MANIFEST_PATH = (
    OPTIMISATION_DATA_DIRECTORY
    / "v05_export_manifest.csv"
)


@st.cache_data
def load_optimisation_manifest():
    """
    Load the v0.5 exported-dataset manifest.
    """

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing optimisation manifest: {MANIFEST_PATH}"
        )

    return pd.read_csv(
        MANIFEST_PATH
    )


@st.cache_data
def load_optimisation_dataset(
    dataset_name,
    index_col=0,
):
    """
    Load one precomputed v0.5 research dataset.
    """

    dataset_path = (
        OPTIMISATION_DATA_DIRECTORY
        / f"{dataset_name}.csv"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing optimisation dataset: {dataset_path}"
        )

    return pd.read_csv(
        dataset_path,
        index_col=index_col,
    )


@st.cache_data
def load_v05_dashboard_data():
    """
    Load the principal datasets used by the v0.5 dashboard.
    """

    dataset_names = [
        "v05_research_summary",
        "optimised_portfolio_weights",
        "optimisation_model_estimates",
        "rolling_performance_comparison",
        "rolling_trading_summary",
        "efficient_frontier",
        "historical_stress_results",
        "stress_test_winners",
        "one_day_risk_summary",
        "monte_carlo_risk_summary",
        "scenario_return_comparison",
        "scenario_stressed_volatility",
        "user_portfolio_optimised_weights",
        "user_portfolio_model_comparison",
        "user_portfolio_monte_carlo",
    ]

    return {
        dataset_name:
            load_optimisation_dataset(
                dataset_name
            )
        for dataset_name in dataset_names
    }
