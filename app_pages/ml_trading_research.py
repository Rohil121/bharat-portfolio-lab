from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.title("🤖 Machine Learning & Trading Research")

st.markdown(
    """
    **Version v0.6 — Walk-Forward ML Research**

    This dashboard presents leakage-safe machine-learning research,
    transaction-cost-adjusted strategy backtests, model comparisons,
    regime-aware allocations and the latest historical notebook signal.
    """
)

st.info(
    "This page presents historical academic research and does not "
    "constitute investment advice or a live trading recommendation."
)


# ---------------------------------------------------------
# Output-file configuration
# ---------------------------------------------------------

REPOSITORY_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

ML_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "outputs"
    / "ml_trading"
)

OUTPUT_FILES = {
    "predictive_models":
        "v06_predictive_model_scorecard.csv",

    "strategies":
        "v06_strategy_scorecard.csv",

    "allocations":
        "v06_latest_portfolio_allocations.csv",

    "portfolio_summary":
        "v06_latest_portfolio_summary.csv",

    "signal_rankings":
        "v06_latest_signal_rankings.csv",

    "conclusions":
        "v06_research_conclusions.csv",
}


# ---------------------------------------------------------
# Load and validate output files
# ---------------------------------------------------------

@st.cache_data
def load_ml_research_outputs():

    missing_files = [
        filename
        for filename in OUTPUT_FILES.values()
        if not (
            ML_OUTPUT_DIRECTORY
            / filename
        ).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "The following v0.6 output files are missing: "
            + ", ".join(
                missing_files
            )
        )

    datasets = {
        dataset_name:
            pd.read_csv(
                ML_OUTPUT_DIRECTORY
                / filename
            )
        for dataset_name, filename in (
            OUTPUT_FILES.items()
        )
    }

    return datasets


try:

    ml_outputs = (
        load_ml_research_outputs()
    )

except Exception as error:

    st.error(
        "The v0.6 research outputs could not be loaded: "
        f"{error}"
    )

    st.stop()


predictive_model_scorecard = (
    ml_outputs[
        "predictive_models"
    ]
    .copy()
)

strategy_scorecard = (
    ml_outputs[
        "strategies"
    ]
    .copy()
)

latest_allocations = (
    ml_outputs[
        "allocations"
    ]
    .copy()
)

latest_portfolio_summary = (
    ml_outputs[
        "portfolio_summary"
    ]
    .copy()
)

latest_signal_rankings = (
    ml_outputs[
        "signal_rankings"
    ]
    .copy()
)

research_conclusions = (
    ml_outputs[
        "conclusions"
    ]
    .copy()
)


# ---------------------------------------------------------
# Validate required columns
# ---------------------------------------------------------

required_columns = {
    "predictive_models": {
        "Model",
        "Model Family",
        "Prediction Task",
    },

    "strategies": {
        "Strategy",
        "CAGR",
        "Sharpe Ratio",
        "Maximum Drawdown",
    },

    "allocations": {
        "Strategy",
        "Signal Date",
        "Execution Date",
        "Ticker",
        "Target Weight",
    },

    "portfolio_summary": {
        "Strategy",
        "Signal Date",
        "Execution Date",
        "Market Regime",
        "Holdings",
    },

    "signal_rankings": {
        "Strategy",
        "Ticker",
        "Score",
        "Signal Rank",
    },

    "conclusions": {
        "Finding",
        "Evidence",
        "Interpretation",
    },
}

dataset_lookup = {
    "predictive_models":
        predictive_model_scorecard,

    "strategies":
        strategy_scorecard,

    "allocations":
        latest_allocations,

    "portfolio_summary":
        latest_portfolio_summary,

    "signal_rankings":
        latest_signal_rankings,

    "conclusions":
        research_conclusions,
}

schema_errors = []

for dataset_name, expected_columns in (
    required_columns.items()
):

    missing_columns = (
        expected_columns
        - set(
            dataset_lookup[
                dataset_name
            ].columns
        )
    )

    if missing_columns:

        schema_errors.append(
            f"{dataset_name}: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

if schema_errors:

    st.error(
        "The v0.6 output schema is incomplete:\n\n"
        + "\n".join(
            schema_errors
        )
    )

    st.stop()


# ---------------------------------------------------------
# Parse dates and identify latest research state
# ---------------------------------------------------------

for dataset in [
    latest_allocations,
    latest_portfolio_summary,
]:

    dataset[
        "Signal Date"
    ] = pd.to_datetime(
        dataset[
            "Signal Date"
        ]
    )

    dataset[
        "Execution Date"
    ] = pd.to_datetime(
        dataset[
            "Execution Date"
        ]
    )


latest_signal_date = (
    latest_portfolio_summary[
        "Signal Date"
    ]
    .max()
)

latest_execution_date = (
    latest_portfolio_summary[
        "Execution Date"
    ]
    .max()
)

latest_market_regime = (
    latest_portfolio_summary[
        "Market Regime"
    ]
    .dropna()
    .iloc[0]
)

regime_aware_row = (
    strategy_scorecard.loc[
        strategy_scorecard[
            "Strategy"
        ]
        == "Regime-Aware Walk-Forward"
    ]
)

if regime_aware_row.empty:

    st.error(
        "The regime-aware strategy is missing from "
        "the final strategy scorecard."
    )

    st.stop()


regime_aware_row = (
    regime_aware_row.iloc[0]
)


# ---------------------------------------------------------
# Initial page status
# ---------------------------------------------------------

metric_columns = st.columns(
    4
)

metric_columns[0].metric(
    "Predictive Methods",
    f"{predictive_model_scorecard['Model'].nunique()}",
)

metric_columns[1].metric(
    "Backtested Strategies",
    f"{strategy_scorecard['Strategy'].nunique()}",
)

metric_columns[2].metric(
    "Latest Signal",
    latest_signal_date.strftime(
        "%d %b %Y"
    ),
)

metric_columns[3].metric(
    "Market Regime",
    latest_market_regime,
)


st.success(
    "All six v0.6 research datasets loaded and passed "
    "the initial application schema validation."
)


st.subheader(
    "Flagship Historical Result"
)

flagship_columns = st.columns(
    4
)

flagship_columns[0].metric(
    "Regime-Aware CAGR",
    f"{regime_aware_row['CAGR']:.2%}",
)

flagship_columns[1].metric(
    "Sharpe Ratio",
    f"{regime_aware_row['Sharpe Ratio']:.2f}",
)

flagship_columns[2].metric(
    "Maximum Drawdown",
    f"{regime_aware_row['Maximum Drawdown']:.2%}",
)

flagship_columns[3].metric(
    "Execution Date",
    latest_execution_date.strftime(
        "%d %b %Y"
    ),
)


st.caption(
    "The complete interactive tables, model comparisons, "
    "allocations and research conclusions will be added "
    "in the next dashboard step."
)
