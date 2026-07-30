"""
Interactive User Portfolio Lab.

Users can enter arbitrary NSE/BSE holdings and run:
    Historical analytics
    Minimum-volatility optimisation
    Maximum-Sharpe optimisation
    Risk parity
    Efficient frontier
    Historical stress testing
    VaR and Expected Shortfall
    Monte Carlo simulation
    Bull, base and bear scenarios
"""

from __future__ import annotations

from datetime import date
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import streamlit as st

from src.portfolio.optimisation import (
    build_efficient_frontier,
    optimise_user_portfolio,
)

from src.portfolio.risk import (
    analyse_user_portfolio_risk,
)

from src.portfolio.user_portfolio import (
    construct_user_portfolio,
    download_user_market_data,
)


DEFAULT_PORTFOLIO = pd.DataFrame(
    {
        "Ticker": [
            "HDFCBANK",
            "TCS",
            "HINDUNILVR",
            "SUNPHARMA",
            "POWERGRID",
            "BHARTIARTL",
            "LT",
            "M&M",
            "BEL",
            "TRENT",
        ],

        "Exchange": [
            "NSE",
            "NSE",
            "NSE",
            "NSE",
            "NSE",
            "NSE",
            "NSE",
            "NSE",
            "NSE",
            "NSE",
        ],

        "Value": [
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
        ],
    }
)


BENCHMARK_OPTIONS = {
    "Nifty 50": "^NSEI",
    "BSE Sensex": "^BSESN",
    "Custom benchmark": None,
}


def format_percentage(
    value,
    decimals=2,
):
    """
    Format a numeric value as a percentage.
    """

    if pd.isna(value):
        return "—"

    return f"{value:.{decimals}%}"


def prepare_portfolio_records(
    edited_portfolio,
):
    """
    Clean the editable table and return cache-safe records.
    """

    clean_input = (
        edited_portfolio
        .copy()
    )

    clean_input[
        "Ticker"
    ] = (
        clean_input[
            "Ticker"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    clean_input[
        "Exchange"
    ] = (
        clean_input[
            "Exchange"
        ]
        .fillna("NSE")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    clean_input[
        "Value"
    ] = pd.to_numeric(
        clean_input[
            "Value"
        ],
        errors="coerce",
    )

    clean_input = (
        clean_input.loc[
            clean_input[
                "Ticker"
            ]
            != ""
        ]
        .reset_index(
            drop=True
        )
    )

    return tuple(
        (
            str(row["Ticker"]),
            str(row["Exchange"]),
            float(row["Value"]),
        )
        for _, row in (
            clean_input.iterrows()
        )
    )


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def run_user_portfolio_analysis(
    portfolio_records,
    input_mode,
    start_date_text,
    end_date_text,
    benchmark_ticker,
    minimum_weight,
    maximum_weight,
    shrinkage,
    risk_free_rate,
    initial_investment,
    simulations,
):
    """
    Run the full production analytics workflow.
    """

    portfolio_input = pd.DataFrame(
        portfolio_records,
        columns=[
            "Ticker",
            "Exchange",
            "Value",
        ],
    )

    constructed_portfolio = (
        construct_user_portfolio(
            portfolio_input=(
                portfolio_input
            ),
            input_mode=input_mode,
        )
    )

    market_data = (
        download_user_market_data(
            portfolio_weights=(
                constructed_portfolio[
                    "weights"
                ]
            ),
            start_date=start_date_text,
            end_date=end_date_text,
            benchmark=benchmark_ticker,
            minimum_observations=252,
        )
    )

    optimisation = (
        optimise_user_portfolio(
            asset_returns=(
                market_data[
                    "portfolio_returns"
                ]
            ),
            current_weights=(
                market_data[
                    "weights"
                ]
            ),
            benchmark_returns=(
                market_data[
                    "benchmark_returns"
                ]
            ),
            minimum_stock_weight=(
                minimum_weight
            ),
            maximum_stock_weight=(
                maximum_weight
            ),
            expected_return_shrinkage=(
                shrinkage
            ),
            risk_free_rate=(
                risk_free_rate
            ),
            initial_investment_inr=(
                initial_investment
            ),
        )
    )

    frontier = (
        build_efficient_frontier(
            expected_returns=(
                optimisation[
                    "expected_returns"
                ]
            ),
            covariance_matrix=(
                optimisation[
                    "covariance"
                ]
            ),
            comparison_weights=(
                optimisation[
                    "weights"
                ]
            ),
            minimum_stock_weight=(
                minimum_weight
            ),
            maximum_stock_weight=(
                maximum_weight
            ),
            risk_free_rate=(
                risk_free_rate
            ),
            frontier_points=60,
        )
    )

    risk = (
        analyse_user_portfolio_risk(
            asset_returns=(
                market_data[
                    "portfolio_returns"
                ]
            ),
            benchmark_returns=(
                market_data[
                    "benchmark_returns"
                ]
            ),
            portfolio_weights=(
                optimisation[
                    "weights"
                ]
            ),
            expected_returns=(
                optimisation[
                    "expected_returns"
                ]
            ),
            covariance_matrix=(
                optimisation[
                    "covariance"
                ]
            ),
            initial_value_inr=(
                initial_investment
            ),
            simulations=simulations,
            horizon_days=252,
            block_length=21,
            estimation_window=756,
            random_seed=42,
        )
    )

    return {
        "portfolio":
            constructed_portfolio,

        "market_data":
            market_data,

        "optimisation":
            optimisation,

        "frontier":
            frontier,

        "risk":
            risk,

        "settings":
            {
                "Benchmark":
                    benchmark_ticker,

                "Start Date":
                    start_date_text,

                "End Date":
                    end_date_text,

                "Risk-Free Rate":
                    risk_free_rate,

                "Expected Return Shrinkage":
                    shrinkage,

                "Minimum Weight":
                    minimum_weight,

                "Maximum Weight":
                    maximum_weight,

                "Initial Investment (₹)":
                    initial_investment,

                "Monte Carlo Simulations":
                    simulations,
            },
    }


def create_results_zip(
    analysis,
):
    """
    Package the principal user results into a downloadable ZIP.
    """

    datasets = {
        "portfolio_input":
            analysis[
                "portfolio"
            ][
                "portfolio_table"
            ],

        "data_quality":
            analysis[
                "market_data"
            ][
                "data_quality"
            ],

        "optimised_weights":
            analysis[
                "optimisation"
            ][
                "weights"
            ],

        "model_comparison":
            analysis[
                "optimisation"
            ][
                "model_comparison"
            ],

        "historical_comparison":
            analysis[
                "optimisation"
            ][
                "realised_comparison"
            ],

        "efficient_frontier":
            analysis[
                "frontier"
            ][
                "frontier"
            ],

        "efficient_frontier_weights":
            analysis[
                "frontier"
            ][
                "frontier_weights"
            ],

        "best_frontier_weights":
            analysis[
                "frontier"
            ][
                "best_frontier_weights"
            ],

        "diversification":
            analysis[
                "risk"
            ][
                "diversification"
            ],

        "one_day_var_es":
            analysis[
                "risk"
            ][
                "one_day_risk"
            ],

        "stress_periods":
            analysis[
                "risk"
            ][
                "stress_periods"
            ],

        "stress_results":
            analysis[
                "risk"
            ][
                "stress_results"
            ],

        "stress_winners":
            analysis[
                "risk"
            ][
                "stress_winners"
            ],

        "monte_carlo_summary":
            analysis[
                "risk"
            ][
                "monte_carlo"
            ],

        "scenario_results":
            analysis[
                "risk"
            ][
                "scenario_results"
            ],

        "scenario_returns":
            analysis[
                "risk"
            ][
                "scenario_return_table"
            ],

        "scenario_volatility":
            analysis[
                "risk"
            ][
                "scenario_volatility_table"
            ],

        "analysis_settings":
            pd.Series(
                analysis[
                    "settings"
                ],
                name="Value",
            ),
    }

    output = BytesIO()

    with ZipFile(
        output,
        mode="w",
        compression=ZIP_DEFLATED,
    ) as zip_file:

        for dataset_name, dataset in (
            datasets.items()
        ):

            if isinstance(
                dataset,
                pd.Series,
            ):
                export_table = (
                    dataset.to_frame()
                )

            else:
                export_table = (
                    dataset.copy()
                )

            zip_file.writestr(
                f"{dataset_name}.csv",
                export_table.to_csv(
                    index=True
                ),
            )

    output.seek(0)

    return output.getvalue()


def render_portfolio_overview(
    analysis,
):
    """
    Render user-input and headline performance results.
    """

    portfolio = analysis[
        "portfolio"
    ]

    market_data = analysis[
        "market_data"
    ]

    realised = analysis[
        "optimisation"
    ][
        "realised_comparison"
    ]

    model = analysis[
        "optimisation"
    ][
        "model_comparison"
    ]

    metric_1, metric_2, metric_3, metric_4 = (
        st.columns(4)
    )

    metric_1.metric(
        "Holdings",
        int(
            portfolio[
                "summary"
            ][
                "Number of Holdings"
            ]
        ),
    )

    metric_2.metric(
        "Aligned observations",
        len(
            market_data[
                "portfolio_returns"
            ]
        ),
    )

    metric_3.metric(
        "Current portfolio CAGR",
        format_percentage(
            realised.loc[
                "Current Portfolio",
                "CAGR",
            ]
        ),
    )

    metric_4.metric(
        "Maximum-Sharpe model return",
        format_percentage(
            model.loc[
                "Maximum Sharpe",
                "Expected Return",
            ]
        ),
    )

    st.markdown("### Normalised portfolio")

    portfolio_table = (
        portfolio[
            "portfolio_table"
        ]
        .copy()
    )

    display_columns = [
        column
        for column in [
            "Standardised Ticker",
            "Value",
            "Quantity",
            "Latest Price (₹)",
            "Market Value (₹)",
            "Calculated Weight",
        ]
        if column in portfolio_table.columns
    ]

    st.dataframe(
        portfolio_table[
            display_columns
        ].style.format(
            {
                "Value": "{:,.2f}",
                "Quantity": "{:,.2f}",
                "Latest Price (₹)": "₹{:,.2f}",
                "Market Value (₹)": "₹{:,.2f}",
                "Calculated Weight": "{:.2%}",
            },
            na_rep="—",
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Market-data quality")

    st.dataframe(
        market_data[
            "data_quality"
        ].style.format(
            {
                "Missing Percentage":
                    "{:.2%}",
            },
            na_rep="—",
        ),
        width="stretch",
    )


def render_optimisation_results(
    analysis,
):
    """
    Render portfolio optimisation results.
    """

    optimisation = analysis[
        "optimisation"
    ]

    st.markdown("### Portfolio weights")

    st.dataframe(
        optimisation[
            "weights"
        ].style.format(
            "{:.2%}",
            na_rep="—",
        ),
        width="stretch",
    )

    st.markdown("### Model-implied comparison")

    st.dataframe(
        optimisation[
            "model_comparison"
        ].style.format(
            {
                "Expected Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe Ratio": "{:.2f}",
                "Largest Weight": "{:.2%}",
                "Active Holdings": "{:.0f}",
                "Largest Risk Contribution": "{:.2%}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.markdown("### Historical comparison")

    st.dataframe(
        optimisation[
            "realised_comparison"
        ].style.format(
            {
                "Observations": "{:,.0f}",
                "Ending Value (₹)": "₹{:,.0f}",
                "Total Return": "{:.2%}",
                "CAGR": "{:.2%}",
                "Annualised Volatility": "{:.2%}",
                "Sharpe Ratio": "{:.2f}",
                "Sortino Ratio": "{:.2f}",
                "Maximum Drawdown": "{:.2%}",
                "Calmar Ratio": "{:.2f}",
                "Beta vs Benchmark": "{:.2f}",
                "Correlation vs Benchmark": "{:.2f}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.warning(
        "Historical results are in-sample comparisons and should "
        "not be interpreted as evidence of future performance."
    )


def render_frontier_results(
    analysis,
):
    """
    Render the constrained efficient frontier.
    """

    frontier = analysis[
        "frontier"
    ]

    frontier_table = (
        frontier[
            "frontier"
        ]
        .reset_index()
    )

    chart_data = (
        frontier_table[
            [
                "Volatility",
                "Expected Return",
            ]
        ]
    )

    st.scatter_chart(
        chart_data,
        x="Volatility",
        y="Expected Return",
        width="stretch",
    )

    metric_1, metric_2, metric_3 = (
        st.columns(3)
    )

    best_result = (
        frontier[
            "best_frontier_result"
        ]
    )

    metric_1.metric(
        "Best frontier return",
        format_percentage(
            best_result[
                "Expected Return"
            ]
        ),
    )

    metric_2.metric(
        "Best frontier volatility",
        format_percentage(
            best_result[
                "Volatility"
            ]
        ),
    )

    metric_3.metric(
        "Best frontier Sharpe",
        f"{best_result['Sharpe Ratio']:.2f}",
    )

    st.markdown("### Best frontier weights")

    st.dataframe(
        frontier[
            "best_frontier_weights"
        ]
        .sort_values(
            ascending=False
        )
        .to_frame(
            "Weight"
        )
        .style.format(
            "{:.2%}"
        ),
        width="stretch",
    )

    st.markdown("### Portfolio comparison points")

    st.dataframe(
        frontier[
            "comparison_points"
        ].style.format(
            {
                "Expected Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe Ratio": "{:.2f}",
            },
            na_rep="—",
        ),
        width="stretch",
    )


def render_risk_results(
    analysis,
):
    """
    Render diversification, VaR, ES and Monte Carlo risk.
    """

    risk = analysis[
        "risk"
    ]

    st.markdown("### Diversification and concentration")

    st.dataframe(
        risk[
            "diversification"
        ].style.format(
            {
                "Number of Holdings": "{:.0f}",
                "Active Holdings": "{:.0f}",
                "Largest Holding": "{:.2%}",
                "Top-3 Concentration": "{:.2%}",
                "Herfindahl Index": "{:.4f}",
                "Effective Holdings": "{:.2f}",
                "Average Pairwise Correlation": "{:.2f}",
                "Diversification Ratio": "{:.2f}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.markdown("### One-day VaR and Expected Shortfall")

    st.dataframe(
        risk[
            "one_day_risk"
        ].style.format(
            {
                "1-Day VaR 95%": "{:.2%}",
                "1-Day ES 95%": "{:.2%}",
                "1-Day VaR 99%": "{:.2%}",
                "1-Day ES 99%": "{:.2%}",
                "Worst Historical Day": "{:.2%}",
                "95% VaR Amount (₹)": "₹{:,.0f}",
                "95% ES Amount (₹)": "₹{:,.0f}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.markdown("### One-year Monte Carlo risk")

    st.dataframe(
        risk[
            "monte_carlo"
        ].style.format(
            {
                "Mean Terminal Return": "{:.2%}",
                "Median Terminal Return": "{:.2%}",
                "5th Percentile Return": "{:.2%}",
                "95th Percentile Return": "{:.2%}",
                "Probability of Loss": "{:.2%}",
                "Probability of Loss > 10%": "{:.2%}",
                "Probability of Loss > 20%": "{:.2%}",
                "One-Year VaR 95%": "{:.2%}",
                "One-Year ES 95%": "{:.2%}",
                "One-Year VaR 99%": "{:.2%}",
                "One-Year ES 99%": "{:.2%}",
                "Median Maximum Drawdown": "{:.2%}",
                "5th Percentile Maximum Drawdown": "{:.2%}",
                "Median Terminal Value (₹)": "₹{:,.0f}",
                "5th Percentile Terminal Value (₹)": "₹{:,.0f}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.caption(
        "Monte Carlo results are model-based risk estimates, "
        "not forecasts or guarantees."
    )


def render_stress_and_scenarios(
    analysis,
):
    """
    Render historical stress and scenario analysis.
    """

    risk = analysis[
        "risk"
    ]

    st.markdown("### Identified historical stress periods")

    st.dataframe(
        risk[
            "stress_periods"
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Historical stress-test winners")

    st.dataframe(
        risk[
            "stress_winners"
        ].style.format(
            {
                "Best Portfolio Return": "{:.2%}",
                "Benchmark Return": "{:.2%}",
                "Loss Reduction vs Benchmark": "{:+.2%}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.markdown("### Bull, base and bear returns")

    st.dataframe(
        risk[
            "scenario_return_table"
        ].style.format(
            "{:.2%}",
            na_rep="—",
        ),
        width="stretch",
    )

    st.markdown("### Scenario volatility")

    st.dataframe(
        risk[
            "scenario_volatility_table"
        ].style.format(
            "{:.2%}",
            na_rep="—",
        ),
        width="stretch",
    )

    st.caption(
        "Scenario results are assumption-based estimates and "
        "are not predictions of future market returns."
    )


def render_user_portfolio_lab():
    """
    Render the complete interactive application page.
    """

    st.title(
        "🧮 User Portfolio Lab"
    )

    st.markdown(
        """
        Build and analyse your own Indian listed-equity portfolio.

        Enter NSE or BSE stocks using **weights**, **invested
        amounts**, or **quantities**, then run the complete
        optimisation and risk workflow.
        """
    )

    st.info(
        "Market data are downloaded on demand. Prices may be "
        "delayed and depend on the third-party data source."
    )

    with st.form(
        "user_portfolio_form"
    ):

        st.markdown(
            "### 1. Enter portfolio holdings"
        )

        input_mode = st.selectbox(
            "Input mode",
            [
                "Weight",
                "Invested Amount",
                "Quantity",
            ],
            index=0,
            help=(
                "Weight values are normalised automatically. "
                "Invested amounts are entered in ₹. Quantity "
                "represents the number of shares."
            ),
        )

        edited_portfolio = st.data_editor(
            DEFAULT_PORTFOLIO,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "Ticker":
                    st.column_config.TextColumn(
                        "Ticker",
                        help=(
                            "Enter the NSE/BSE symbol without "
                            "the .NS or .BO suffix."
                        ),
                        required=True,
                    ),

                "Exchange":
                    st.column_config.SelectboxColumn(
                        "Exchange",
                        options=[
                            "NSE",
                            "BSE",
                        ],
                        required=True,
                    ),

                "Value":
                    st.column_config.NumberColumn(
                        "Value",
                        min_value=0.0001,
                        format="%.2f",
                        required=True,
                    ),
            },
        )

        st.markdown(
            "### 2. Select market assumptions"
        )

        column_1, column_2, column_3 = (
            st.columns(3)
        )

        with column_1:

            start_date = st.date_input(
                "Start date",
                value=date(
                    2021,
                    4,
                    1,
                ),
            )

            end_date = st.date_input(
                "End date",
                value=date.today(),
            )

        with column_2:

            benchmark_name = st.selectbox(
                "Benchmark",
                list(
                    BENCHMARK_OPTIONS.keys()
                ),
                index=0,
            )

            custom_benchmark = st.text_input(
                "Custom benchmark ticker",
                value="",
                disabled=(
                    benchmark_name
                    != "Custom benchmark"
                ),
                help=(
                    "Example: ^NSEI or ^BSESN."
                ),
            )

            risk_free_rate_percent = (
                st.number_input(
                    "Risk-free rate (%)",
                    min_value=0.0,
                    max_value=25.0,
                    value=6.5,
                    step=0.1,
                )
            )

        with column_3:

            initial_investment = (
                st.number_input(
                    "Initial investment (₹)",
                    min_value=10_000,
                    value=1_000_000,
                    step=50_000,
                )
            )

            simulations = st.selectbox(
                "Monte Carlo simulations",
                [
                    1_000,
                    2_000,
                    5_000,
                    10_000,
                ],
                index=1,
            )

        st.markdown(
            "### 3. Set optimisation constraints"
        )

        constraint_1, constraint_2, constraint_3 = (
            st.columns(3)
        )

        with constraint_1:

            minimum_weight_percent = (
                st.number_input(
                    "Minimum stock weight (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=0.0,
                    step=1.0,
                )
            )

        with constraint_2:

            maximum_weight_percent = (
                st.number_input(
                    "Maximum stock weight (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=30.0,
                    step=1.0,
                )
            )

        with constraint_3:

            shrinkage_percent = (
                st.number_input(
                    "Expected-return shrinkage (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    step=5.0,
                )
            )

        submitted = st.form_submit_button(
            "Run Full Portfolio Analysis",
            type="primary",
            width="stretch",
        )

    if submitted:

        try:

            portfolio_records = (
                prepare_portfolio_records(
                    edited_portfolio
                )
            )

            if len(portfolio_records) < 2:
                raise ValueError(
                    "Enter at least two valid holdings."
                )

            if (
                benchmark_name
                == "Custom benchmark"
            ):

                benchmark_ticker = (
                    custom_benchmark
                    .strip()
                    .upper()
                )

                if not benchmark_ticker:
                    raise ValueError(
                        "Enter a custom benchmark ticker."
                    )

            else:

                benchmark_ticker = (
                    BENCHMARK_OPTIONS[
                        benchmark_name
                    ]
                )

            with st.spinner(
                "Downloading market data and running the "
                "complete portfolio analysis..."
            ):

                analysis = (
                    run_user_portfolio_analysis(
                        portfolio_records=(
                            portfolio_records
                        ),
                        input_mode=input_mode,
                        start_date_text=(
                            start_date.isoformat()
                        ),
                        end_date_text=(
                            end_date.isoformat()
                        ),
                        benchmark_ticker=(
                            benchmark_ticker
                        ),
                        minimum_weight=(
                            minimum_weight_percent
                            / 100.0
                        ),
                        maximum_weight=(
                            maximum_weight_percent
                            / 100.0
                        ),
                        shrinkage=(
                            shrinkage_percent
                            / 100.0
                        ),
                        risk_free_rate=(
                            risk_free_rate_percent
                            / 100.0
                        ),
                        initial_investment=float(
                            initial_investment
                        ),
                        simulations=int(
                            simulations
                        ),
                    )
                )

            st.session_state[
                "user_portfolio_analysis"
            ] = analysis

            st.success(
                "Portfolio analysis completed successfully."
            )

        except Exception as error:

            st.error(
                "The portfolio analysis could not be completed."
            )

            st.exception(
                error
            )

    analysis = st.session_state.get(
        "user_portfolio_analysis"
    )

    if analysis is None:

        st.markdown("---")

        st.caption(
            "Complete the form and run the analysis to view "
            "optimisation, stress-testing and simulation results."
        )

        return

    st.markdown("---")

    (
        overview_tab,
        optimisation_tab,
        frontier_tab,
        risk_tab,
        stress_tab,
        download_tab,
    ) = st.tabs(
        [
            "Portfolio Overview",
            "Optimisation",
            "Efficient Frontier",
            "Risk Analysis",
            "Stress & Scenarios",
            "Downloads",
        ]
    )

    with overview_tab:
        render_portfolio_overview(
            analysis
        )

    with optimisation_tab:
        render_optimisation_results(
            analysis
        )

    with frontier_tab:
        render_frontier_results(
            analysis
        )

    with risk_tab:
        render_risk_results(
            analysis
        )

    with stress_tab:
        render_stress_and_scenarios(
            analysis
        )

    with download_tab:

        st.markdown(
            "### Download complete analysis"
        )

        st.write(
            "The ZIP file contains portfolio weights, "
            "optimisation results, efficient-frontier data, "
            "VaR, Expected Shortfall, stress tests, Monte Carlo "
            "results and scenario analysis."
        )

        st.download_button(
            label="Download Portfolio Analysis ZIP",
            data=create_results_zip(
                analysis
            ),
            file_name=(
                "bharat_portfolio_lab_results.zip"
            ),
            mime="application/zip",
            type="primary",
        )

    st.caption(
        "Bharat Portfolio Lab is an academic analytics project "
        "and does not constitute investment advice."
    )


render_user_portfolio_lab()
