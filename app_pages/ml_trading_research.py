from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# PAGE HEADER
# =========================================================

st.title("🤖 Machine Learning & Trading Research")

st.markdown(
    """
    **Version v0.6 — Walk-Forward Research for Indian Equities**

    This page compares rule-based signals, machine-learning models,
    portfolio-construction methods and a dynamic market-regime selector
    using leakage-safe historical backtests.
    """
)

st.warning(
    "Historical research only. The displayed signals and allocations "
    "are not live investment recommendations and do not guarantee "
    "future performance."
)


# =========================================================
# FILE CONFIGURATION
# =========================================================

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


# =========================================================
# DATA LOADING
# =========================================================

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
            "Missing v0.6 output files: "
            + ", ".join(
                missing_files
            )
        )

    return {
        dataset_name:
            pd.read_csv(
                ML_OUTPUT_DIRECTORY
                / filename
            )
        for dataset_name, filename in (
            OUTPUT_FILES.items()
        )
    }


try:

    ml_outputs = load_ml_research_outputs()

except Exception as error:

    st.error(
        "The v0.6 research outputs could not be loaded."
    )

    st.code(
        str(error)
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


# =========================================================
# SCHEMA VALIDATION
# =========================================================

required_columns = {
    "predictive_models": {
        "Model",
        "Model Family",
        "Prediction Task",
    },

    "strategies": {
        "Strategy",
        "Strategy Type",
        "Ending Value (₹)",
        "CAGR",
        "Annualised Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Annualised Turnover",
    },

    "allocations": {
        "Strategy",
        "Signal Date",
        "Execution Date",
        "Ticker",
        "Target Weight",
        "Selected",
        "Market Regime",
    },

    "portfolio_summary": {
        "Strategy",
        "Signal Date",
        "Execution Date",
        "Market Regime",
        "Source Strategy",
        "Construction Method",
        "Active Holdings",
        "Largest Weight",
        "Holdings",
    },

    "signal_rankings": {
        "Date",
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
        "The v0.6 output schema is incomplete."
    )

    st.code(
        "\n".join(
            schema_errors
        )
    )

    st.stop()


# =========================================================
# DATA CLEANING
# =========================================================

for dataset in [
    latest_allocations,
    latest_portfolio_summary,
]:

    dataset[
        "Signal Date"
    ] = pd.to_datetime(
        dataset[
            "Signal Date"
        ],
        errors="coerce",
    )

    dataset[
        "Execution Date"
    ] = pd.to_datetime(
        dataset[
            "Execution Date"
        ],
        errors="coerce",
    )


latest_signal_rankings[
    "Date"
] = pd.to_datetime(
    latest_signal_rankings[
        "Date"
    ],
    errors="coerce",
)


boolean_columns = [
    "Selected",
]

for column in boolean_columns:

    if column in latest_allocations.columns:

        latest_allocations[
            column
        ] = (
            latest_allocations[
                column
            ]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
            .fillna(
                False
            )
        )


if "Selected" in (
    latest_signal_rankings.columns
):

    latest_signal_rankings[
        "Selected"
    ] = (
        latest_signal_rankings[
            "Selected"
        ]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(
            False
        )
    )


numeric_strategy_columns = [
    "Ending Value (₹)",
    "Total Return",
    "CAGR",
    "Annualised Volatility",
    "Sharpe Ratio",
    "Maximum Drawdown",
    "Calmar Ratio",
    "Beta vs Nifty",
    "Correlation vs Nifty",
    "Annualised Turnover",
    "Transaction Costs (₹)",
]

for column in numeric_strategy_columns:

    if column in strategy_scorecard.columns:

        strategy_scorecard[
            column
        ] = pd.to_numeric(
            strategy_scorecard[
                column
            ],
            errors="coerce",
        )


predictive_numeric_columns = [
    "Mean Rank IC",
    "Median Rank IC",
    "Positive Rank IC Rate",
    "Mean Selected Forward Return",
    "Mean Selected Excess Return",
    "Selected Positive Return Rate",
    "Selected Outperformance Rate",
    "Mean Absolute Error",
    "Root Mean Squared Error",
    "Pooled R-Squared",
    "Directional Accuracy",
    "Accuracy",
    "Balanced Accuracy",
    "ROC AUC",
    "Brier Score",
    "CAGR",
    "Annualised Volatility",
    "Sharpe Ratio",
    "Maximum Drawdown",
    "Annualised Turnover",
]

for column in predictive_numeric_columns:

    if column in (
        predictive_model_scorecard.columns
    ):

        predictive_model_scorecard[
            column
        ] = pd.to_numeric(
            predictive_model_scorecard[
                column
            ],
            errors="coerce",
        )


latest_allocations[
    "Target Weight"
] = pd.to_numeric(
    latest_allocations[
        "Target Weight"
    ],
    errors="coerce",
)

latest_signal_rankings[
    "Score"
] = pd.to_numeric(
    latest_signal_rankings[
        "Score"
    ],
    errors="coerce",
)

latest_signal_rankings[
    "Signal Rank"
] = pd.to_numeric(
    latest_signal_rankings[
        "Signal Rank"
    ],
    errors="coerce",
)


# =========================================================
# RESEARCH STATE
# =========================================================

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
    .iloc[
        0
    ]
)

regime_strategy_name = (
    "Regime-Aware Walk-Forward"
)

regime_aware_rows = (
    strategy_scorecard.loc[
        strategy_scorecard[
            "Strategy"
        ]
        == regime_strategy_name
    ]
)

if regime_aware_rows.empty:

    st.error(
        "The regime-aware strategy is missing "
        "from the strategy scorecard."
    )

    st.stop()


regime_aware_result = (
    regime_aware_rows.iloc[
        0
    ]
)

regime_portfolio_rows = (
    latest_portfolio_summary.loc[
        latest_portfolio_summary[
            "Strategy"
        ]
        == regime_strategy_name
    ]
)

if regime_portfolio_rows.empty:

    st.error(
        "The latest regime-aware allocation is missing."
    )

    st.stop()


regime_portfolio_result = (
    regime_portfolio_rows.iloc[
        0
    ]
)

latest_selected_strategy = (
    regime_portfolio_result[
        "Source Strategy"
    ]
)


# =========================================================
# FORMATTING HELPERS
# =========================================================

def format_inr(
    value,
):

    if pd.isna(
        value
    ):
        return "—"

    value = float(
        value
    )

    if abs(
        value
    ) >= 10_000_000:

        return (
            f"₹{value / 10_000_000:,.2f} Cr"
        )

    if abs(
        value
    ) >= 100_000:

        return (
            f"₹{value / 100_000:,.2f} L"
        )

    return (
        f"₹{value:,.0f}"
    )


def format_percentage(
    value,
    decimal_places=2,
):

    if pd.isna(
        value
    ):
        return "—"

    return (
        f"{float(value):.{decimal_places}%}"
    )


def safe_date_text(
    value,
):

    if pd.isna(
        value
    ):
        return "—"

    return pd.Timestamp(
        value
    ).strftime(
        "%d %b %Y"
    )


def create_csv_bytes(
    dataframe,
    include_index=False,
):

    return (
        dataframe.to_csv(
            index=include_index
        )
        .encode(
            "utf-8"
        )
    )


# =========================================================
# TOP-LEVEL METRICS
# =========================================================

top_metric_columns = st.columns(
    5
)

top_metric_columns[
    0
].metric(
    "Predictive Methods",
    predictive_model_scorecard[
        "Model"
    ].nunique(),
)

top_metric_columns[
    1
].metric(
    "Backtested Strategies",
    strategy_scorecard[
        "Strategy"
    ].nunique(),
)

top_metric_columns[
    2
].metric(
    "Latest Signal",
    safe_date_text(
        latest_signal_date
    ),
)

top_metric_columns[
    3
].metric(
    "Market Regime",
    latest_market_regime,
)

top_metric_columns[
    4
].metric(
    "Selected Strategy",
    latest_selected_strategy,
)


st.caption(
    "Latest signal and allocation refer to the final "
    "historical date in the v0.6 research dataset."
)


# =========================================================
# DASHBOARD TABS
# =========================================================

(
    overview_tab,
    leaderboard_tab,
    portfolio_tab,
    model_tab,
    signal_tab,
    evidence_tab,
) = st.tabs(
    [
        "Overview",
        "Strategy Leaderboard",
        "Latest Portfolios",
        "Predictive Models",
        "Signal Rankings",
        "Evidence & Conclusions",
    ]
)


# =========================================================
# TAB 1 — OVERVIEW
# =========================================================

with overview_tab:

    st.subheader(
        "Flagship Historical Result"
    )

    flagship_columns = st.columns(
        5
    )

    flagship_columns[
        0
    ].metric(
        "Ending Value",
        format_inr(
            regime_aware_result[
                "Ending Value (₹)"
            ]
        ),
        help=(
            "Historical ending value of an initial "
            "₹10,00,000 investment."
        ),
    )

    flagship_columns[
        1
    ].metric(
        "CAGR",
        format_percentage(
            regime_aware_result[
                "CAGR"
            ]
        ),
    )

    flagship_columns[
        2
    ].metric(
        "Sharpe Ratio",
        (
            f"{regime_aware_result['Sharpe Ratio']:.2f}"
        ),
    )

    flagship_columns[
        3
    ].metric(
        "Maximum Drawdown",
        format_percentage(
            regime_aware_result[
                "Maximum Drawdown"
            ]
        ),
    )

    flagship_columns[
        4
    ].metric(
        "Annual Turnover",
        format_percentage(
            regime_aware_result[
                "Annualised Turnover"
            ]
        ),
    )


    st.markdown(
        "#### Latest regime-aware historical allocation"
    )

    allocation_columns = st.columns(
        4
    )

    allocation_columns[
        0
    ].metric(
        "Signal Date",
        safe_date_text(
            latest_signal_date
        ),
    )

    allocation_columns[
        1
    ].metric(
        "Execution Date",
        safe_date_text(
            latest_execution_date
        ),
    )

    allocation_columns[
        2
    ].metric(
        "Market Regime",
        latest_market_regime,
    )

    allocation_columns[
        3
    ].metric(
        "Chosen Strategy",
        latest_selected_strategy,
    )


    st.markdown(
        f"**Holdings:** "
        f"{regime_portfolio_result['Holdings']}"
    )


    st.divider()

    st.markdown(
        "#### What the v0.6 research found"
    )

    overview_conclusions = (
        research_conclusions.head(
            5
        )
    )

    for _, conclusion in (
        overview_conclusions.iterrows()
    ):

        with st.container(
            border=True
        ):

            st.markdown(
                f"**{conclusion['Finding']}**"
            )

            st.write(
                conclusion[
                    "Evidence"
                ]
            )

            st.caption(
                conclusion[
                    "Interpretation"
                ]
            )


# =========================================================
# TAB 2 — STRATEGY LEADERBOARD
# =========================================================

with leaderboard_tab:

    st.subheader(
        "Transaction-Cost-Adjusted Strategy Comparison"
    )

    st.caption(
        "All strategies use the same historical period, "
        "₹10,00,000 starting capital and 0.15% one-way "
        "transaction-cost assumption where applicable."
    )

    strategy_types = sorted(
        strategy_scorecard[
            "Strategy Type"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    selected_strategy_types = (
        st.multiselect(
            "Strategy types",
            options=strategy_types,
            default=strategy_types,
        )
    )

    sort_metric = st.selectbox(
        "Rank strategies by",
        options=[
            "Sharpe Ratio",
            "CAGR",
            "Ending Value (₹)",
            "Maximum Drawdown",
            "Annualised Turnover",
        ],
        index=0,
    )

    filtered_strategy_scorecard = (
        strategy_scorecard.loc[
            strategy_scorecard[
                "Strategy Type"
            ]
            .isin(
                selected_strategy_types
            )
        ]
        .copy()
    )

    sort_ascending = (
        sort_metric
        in [
            "Annualised Turnover",
        ]
    )

    filtered_strategy_scorecard = (
        filtered_strategy_scorecard
        .sort_values(
            sort_metric,
            ascending=sort_ascending,
        )
    )


    chart_metric = st.radio(
        "Chart metric",
        options=[
            "CAGR",
            "Sharpe Ratio",
        ],
        horizontal=True,
    )

    strategy_chart = (
        filtered_strategy_scorecard[
            [
                "Strategy",
                chart_metric,
            ]
        ]
        .dropna()
        .set_index(
            "Strategy"
        )
    )

    if chart_metric == "CAGR":

        strategy_chart = (
            strategy_chart
            * 100
        )

    st.bar_chart(
        strategy_chart
    )


    strategy_display = (
        filtered_strategy_scorecard[
            [
                "Strategy",
                "Strategy Type",
                "Ending Value (₹)",
                "CAGR",
                "Annualised Volatility",
                "Sharpe Ratio",
                "Maximum Drawdown",
                "Annualised Turnover",
                "Transaction Costs (₹)",
            ]
        ]
        .copy()
    )

    strategy_display[
        "Ending Value (₹)"
    ] = (
        strategy_display[
            "Ending Value (₹)"
        ]
        .map(
            format_inr
        )
    )

    strategy_display[
        "Transaction Costs (₹)"
    ] = (
        strategy_display[
            "Transaction Costs (₹)"
        ]
        .map(
            format_inr
        )
    )

    for column in [
        "CAGR",
        "Annualised Volatility",
        "Maximum Drawdown",
        "Annualised Turnover",
    ]:

        strategy_display[
            column
        ] = (
            strategy_display[
                column
            ]
            .map(
                format_percentage
            )
        )

    strategy_display[
        "Sharpe Ratio"
    ] = (
        strategy_display[
            "Sharpe Ratio"
        ]
        .map(
            lambda value:
                (
                    f"{value:.2f}"
                    if pd.notna(
                        value
                    )
                    else "—"
                )
        )
    )


    st.dataframe(
        strategy_display,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download strategy scorecard",
        data=create_csv_bytes(
            strategy_scorecard
        ),
        file_name=(
            "v06_strategy_scorecard.csv"
        ),
        mime="text/csv",
    )


# =========================================================
# TAB 3 — LATEST PORTFOLIOS
# =========================================================

with portfolio_tab:

    st.subheader(
        "Latest Historical Portfolio Allocations"
    )

    portfolio_strategies = sorted(
        latest_portfolio_summary[
            "Strategy"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    default_portfolio_index = (
        portfolio_strategies.index(
            regime_strategy_name
        )
        if regime_strategy_name
        in portfolio_strategies
        else 0
    )

    selected_portfolio_strategy = (
        st.selectbox(
            "Select strategy",
            options=portfolio_strategies,
            index=default_portfolio_index,
        )
    )

    selected_portfolio_summary = (
        latest_portfolio_summary.loc[
            latest_portfolio_summary[
                "Strategy"
            ]
            == selected_portfolio_strategy
        ]
        .iloc[
            0
        ]
    )

    selected_portfolio_allocations = (
        latest_allocations.loc[
            latest_allocations[
                "Strategy"
            ]
            == selected_portfolio_strategy
        ]
        .copy()
        .sort_values(
            [
                "Target Weight",
                "Ticker",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )


    portfolio_metric_columns = (
        st.columns(
            5
        )
    )

    portfolio_metric_columns[
        0
    ].metric(
        "Signal Date",
        safe_date_text(
            selected_portfolio_summary[
                "Signal Date"
            ]
        ),
    )

    portfolio_metric_columns[
        1
    ].metric(
        "Execution Date",
        safe_date_text(
            selected_portfolio_summary[
                "Execution Date"
            ]
        ),
    )

    portfolio_metric_columns[
        2
    ].metric(
        "Market Regime",
        selected_portfolio_summary[
            "Market Regime"
        ],
    )

    portfolio_metric_columns[
        3
    ].metric(
        "Active Holdings",
        int(
            selected_portfolio_summary[
                "Active Holdings"
            ]
        ),
    )

    portfolio_metric_columns[
        4
    ].metric(
        "Largest Weight",
        format_percentage(
            selected_portfolio_summary[
                "Largest Weight"
            ]
        ),
    )


    st.markdown(
        f"**Construction method:** "
        f"{selected_portfolio_summary['Construction Method']}"
    )

    st.markdown(
        f"**Source strategy:** "
        f"{selected_portfolio_summary['Source Strategy']}"
    )

    st.markdown(
        f"**Selected holdings:** "
        f"{selected_portfolio_summary['Holdings']}"
    )


    positive_allocations = (
        selected_portfolio_allocations.loc[
            selected_portfolio_allocations[
                "Target Weight"
            ]
            > 0
        ]
        .copy()
    )

    allocation_chart = (
        positive_allocations[
            [
                "Ticker",
                "Target Weight",
            ]
        ]
        .set_index(
            "Ticker"
        )
        * 100
    )

    st.bar_chart(
        allocation_chart
    )


    allocation_display = (
        selected_portfolio_allocations[
            [
                "Ticker",
                "Selected",
                "Target Weight",
                "Score",
                "Source Strategy",
                "Construction Method",
            ]
        ]
        .copy()
    )

    allocation_display[
        "Target Weight"
    ] = (
        allocation_display[
            "Target Weight"
        ]
        .map(
            format_percentage
        )
    )

    allocation_display[
        "Score"
    ] = (
        pd.to_numeric(
            allocation_display[
                "Score"
            ],
            errors="coerce",
        )
        .map(
            lambda value:
                (
                    f"{value:.6f}"
                    if pd.notna(
                        value
                    )
                    else "—"
                )
        )
    )


    st.dataframe(
        allocation_display,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download latest allocations",
        data=create_csv_bytes(
            latest_allocations
        ),
        file_name=(
            "v06_latest_portfolio_allocations.csv"
        ),
        mime="text/csv",
    )


# =========================================================
# TAB 4 — PREDICTIVE MODELS
# =========================================================

with model_tab:

    st.subheader(
        "Predictive Model Research"
    )

    st.caption(
        "Prediction quality and portfolio performance "
        "should be assessed separately. A low forecasting "
        "R² does not automatically prevent useful rankings, "
        "while a profitable backtest does not prove stable "
        "predictive skill."
    )

    model_families = sorted(
        predictive_model_scorecard[
            "Model Family"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    selected_model_families = (
        st.multiselect(
            "Model families",
            options=model_families,
            default=model_families,
        )
    )

    filtered_model_scorecard = (
        predictive_model_scorecard.loc[
            predictive_model_scorecard[
                "Model Family"
            ]
            .isin(
                selected_model_families
            )
        ]
        .copy()
    )


    model_metric_columns = st.columns(
        4
    )

    model_metric_columns[
        0
    ].metric(
        "Methods Displayed",
        filtered_model_scorecard[
            "Model"
        ].nunique(),
    )

    model_metric_columns[
        1
    ].metric(
        "Best Mean Rank IC",
        format_percentage(
            filtered_model_scorecard[
                "Mean Rank IC"
            ].max()
        ),
    )

    model_metric_columns[
        2
    ].metric(
        "Best Selected Excess Return",
        format_percentage(
            filtered_model_scorecard[
                "Mean Selected Excess Return"
            ].max()
        ),
    )

    model_metric_columns[
        3
    ].metric(
        "Best Backtested Sharpe",
        (
            f"{filtered_model_scorecard['Sharpe Ratio'].max():.2f}"
        ),
    )


    model_display_columns = [
        "Model",
        "Model Family",
        "Prediction Task",
        "Mean Rank IC",
        "Positive Rank IC Rate",
        "Mean Selected Forward Return",
        "Mean Selected Excess Return",
        "Directional Accuracy",
        "ROC AUC",
        "Pooled R-Squared",
        "CAGR",
        "Sharpe Ratio",
        "Annualised Turnover",
    ]

    model_display_columns = [
        column
        for column in model_display_columns
        if column
        in filtered_model_scorecard.columns
    ]

    model_display = (
        filtered_model_scorecard[
            model_display_columns
        ]
        .copy()
    )

    percentage_model_columns = [
        "Mean Rank IC",
        "Positive Rank IC Rate",
        "Mean Selected Forward Return",
        "Mean Selected Excess Return",
        "Directional Accuracy",
        "ROC AUC",
        "Pooled R-Squared",
        "CAGR",
        "Annualised Turnover",
    ]

    for column in percentage_model_columns:

        if column in model_display.columns:

            model_display[
                column
            ] = (
                model_display[
                    column
                ]
                .map(
                    format_percentage
                )
            )


    if "Sharpe Ratio" in (
        model_display.columns
    ):

        model_display[
            "Sharpe Ratio"
        ] = (
            model_display[
                "Sharpe Ratio"
            ]
            .map(
                lambda value:
                    (
                        f"{value:.2f}"
                        if pd.notna(
                            value
                        )
                        else "—"
                    )
            )
        )


    st.dataframe(
        model_display,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download predictive model scorecard",
        data=create_csv_bytes(
            predictive_model_scorecard
        ),
        file_name=(
            "v06_predictive_model_scorecard.csv"
        ),
        mime="text/csv",
    )


# =========================================================
# TAB 5 — SIGNAL RANKINGS
# =========================================================

with signal_tab:

    st.subheader(
        "Latest Direct Signal Rankings"
    )

    signal_strategies = sorted(
        latest_signal_rankings[
            "Strategy"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    selected_signal_strategy = (
        st.selectbox(
            "Select signal or model",
            options=signal_strategies,
        )
    )

    selected_signal_rankings = (
        latest_signal_rankings.loc[
            latest_signal_rankings[
                "Strategy"
            ]
            == selected_signal_strategy
        ]
        .copy()
        .sort_values(
            "Signal Rank"
        )
    )


    signal_date_value = (
        selected_signal_rankings[
            "Date"
        ]
        .max()
    )

    st.caption(
        "Signal date: "
        + safe_date_text(
            signal_date_value
        )
    )


    signal_chart = (
        selected_signal_rankings[
            [
                "Ticker",
                "Score",
            ]
        ]
        .set_index(
            "Ticker"
        )
    )

    st.bar_chart(
        signal_chart
    )


    signal_display_columns = [
        "Signal Rank",
        "Ticker",
        "Score",
        "Selected",
        "Target Weight",
        "Signal Type",
    ]

    signal_display_columns = [
        column
        for column in signal_display_columns
        if column
        in selected_signal_rankings.columns
    ]

    signal_display = (
        selected_signal_rankings[
            signal_display_columns
        ]
        .copy()
    )

    signal_display[
        "Score"
    ] = (
        signal_display[
            "Score"
        ]
        .map(
            lambda value:
                (
                    f"{value:.6f}"
                    if pd.notna(
                        value
                    )
                    else "—"
                )
        )
    )

    if "Target Weight" in (
        signal_display.columns
    ):

        signal_display[
            "Target Weight"
        ] = (
            pd.to_numeric(
                signal_display[
                    "Target Weight"
                ],
                errors="coerce",
            )
            .map(
                format_percentage
            )
        )


    st.dataframe(
        signal_display,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download latest signal rankings",
        data=create_csv_bytes(
            latest_signal_rankings
        ),
        file_name=(
            "v06_latest_signal_rankings.csv"
        ),
        mime="text/csv",
    )


# =========================================================
# TAB 6 — EVIDENCE AND CONCLUSIONS
# =========================================================

with evidence_tab:

    st.subheader(
        "Research Evidence and Limitations"
    )

    for _, conclusion in (
        research_conclusions.iterrows()
    ):

        with st.expander(
            conclusion[
                "Finding"
            ]
        ):

            st.markdown(
                "**Evidence**"
            )

            st.write(
                conclusion[
                    "Evidence"
                ]
            )

            st.markdown(
                "**Interpretation**"
            )

            st.write(
                conclusion[
                    "Interpretation"
                ]
            )


    st.divider()

    st.markdown(
        "#### Methodology safeguards"
    )

    safeguard_columns = st.columns(
        3
    )

    with safeguard_columns[
        0
    ]:

        st.success(
            "Expanding-window walk-forward evaluation"
        )

        st.success(
            "Training-only feature standardisation"
        )

        st.success(
            "No random train-test split"
        )

    with safeguard_columns[
        1
    ]:

        st.success(
            "One-trading-day execution lag"
        )

        st.success(
            "Monthly rebalancing"
        )

        st.success(
            "Transaction-cost adjustment"
        )

    with safeguard_columns[
        2
    ]:

        st.success(
            "Selector specification robustness"
        )

        st.success(
            "Indian financial-year stability"
        )

        st.success(
            "Block-bootstrap significance audit"
        )


    st.error(
        "Important limitation: the India 10 universe is fixed. "
        "Survivorship bias, stock-universe selection bias and "
        "researcher-choice bias may remain even after walk-forward, "
        "cost, robustness and bootstrap controls."
    )

    st.download_button(
        label="Download research conclusions",
        data=create_csv_bytes(
            research_conclusions
        ),
        file_name=(
            "v06_research_conclusions.csv"
        ),
        mime="text/csv",
    )
