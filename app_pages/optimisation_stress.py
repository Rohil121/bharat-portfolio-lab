import pandas as pd
import streamlit as st

from src.ui.optimisation_data import (
    load_optimisation_manifest,
    load_v05_dashboard_data,
)


def format_percentage_table(
    dataframe,
    percentage_columns,
):
    """
    Return a Streamlit Styler with selected columns as percentages.
    """

    format_map = {
        column: "{:.2%}"
        for column in percentage_columns
        if column in dataframe.columns
    }

    return dataframe.style.format(
        format_map,
        na_rep="—",
    )


def render_research_overview(data):
    """
    Render the principal v0.5 conclusions.
    """

    performance = data[
        "rolling_performance_comparison"
    ]

    maximum_sharpe = performance.loc[
        "Maximum Sharpe"
    ]

    minimum_volatility = performance.loc[
        "Minimum Volatility"
    ]

    risk_parity = performance.loc[
        "Risk Parity"
    ]

    metric_1, metric_2, metric_3, metric_4 = (
        st.columns(4)
    )

    metric_1.metric(
        "Maximum-Sharpe CAGR",
        f"{maximum_sharpe['CAGR']:.2%}",
    )

    metric_2.metric(
        "Maximum-Sharpe Ratio",
        f"{maximum_sharpe['Sharpe Ratio']:.2f}",
    )

    metric_3.metric(
        "Minimum-Volatility Risk",
        (
            f"{minimum_volatility['Annualised Volatility']:.2%}"
        ),
    )

    metric_4.metric(
        "Risk-Parity Drawdown",
        f"{risk_parity['Maximum Drawdown']:.2%}",
    )

    st.markdown("### Research findings")

    st.dataframe(
        data[
            "v05_research_summary"
        ],
        width="stretch",
        hide_index=True,
    )

    st.info(
        "Maximum Sharpe produced the strongest out-of-sample "
        "return and Sharpe ratio, but also carried greater "
        "concentration, turnover and drawdown risk."
    )


def render_optimisation(data):
    """
    Render portfolio weights, model estimates and the frontier.
    """

    st.markdown("## Portfolio optimisation")

    st.markdown("### Portfolio weights")

    weight_data = data[
        "optimised_portfolio_weights"
    ]

    weight_columns = [
        column
        for column in weight_data.columns
        if (
            "Weight" in column
            or column
            in [
                "Current Equal Weight",
                "Minimum Volatility",
                "Maximum Sharpe",
                "Risk Parity",
            ]
        )
    ]

    st.dataframe(
        format_percentage_table(
            weight_data,
            weight_columns,
        ),
        width="stretch",
    )

    st.markdown("### Model-implied estimates")

    model_estimates = data[
        "optimisation_model_estimates"
    ]

    st.dataframe(
        model_estimates.style.format(
            {
                "Model Expected Return": "{:.2%}",
                "Model Volatility": "{:.2%}",
                "Model Sharpe": "{:.2f}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.markdown("### Constrained efficient frontier")

    frontier = (
        data[
            "efficient_frontier"
        ]
        .reset_index()
    )

    st.scatter_chart(
        frontier,
        x="Volatility",
        y="Expected Return",
        width="stretch",
    )

    st.caption(
        "Long-only, fully invested portfolios with a maximum "
        "20% allocation to any single India 10 stock."
    )


def render_out_of_sample(data):
    """
    Render rolling performance and trading-cost evidence.
    """

    st.markdown("## Rolling out-of-sample performance")

    performance = data[
        "rolling_performance_comparison"
    ]

    st.dataframe(
        performance.style.format(
            {
                "Ending Value (₹)": "₹{:,.0f}",
                "Total Return": "{:.2%}",
                "CAGR": "{:.2%}",
                "Annualised Volatility": "{:.2%}",
                "Sharpe Ratio": "{:.2f}",
                "Sortino Ratio": "{:.2f}",
                "Maximum Drawdown": "{:.2%}",
                "Calmar Ratio": "{:.2f}",
                "Beta vs Nifty 50": "{:.2f}",
                "Correlation vs Nifty 50": "{:.2f}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.markdown("### Turnover and transaction costs")

    trading = data[
        "rolling_trading_summary"
    ]

    st.dataframe(
        trading.style.format(
            {
                "Total_One_Way_Turnover": "{:.2%}",
                "Average_Turnover": "{:.2%}",
                "Total_Transaction_Costs_INR": "₹{:,.2f}",
                "Annualised One-Way Turnover": "{:.2%}",
                "Costs as % of Initial Capital": "{:.2%}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.warning(
        "These results use monthly point-in-time estimation, "
        "one-day signal separation and 0.15% transaction costs."
    )


def render_stress_testing(data):
    """
    Render historical stress and drawdown evidence.
    """

    st.markdown("## Historical stress testing")

    winners = data[
        "stress_test_winners"
    ]

    st.dataframe(
        winners.style.format(
            {
                "Best Portfolio Return": "{:.2%}",
                "Nifty 50 Return": "{:.2%}",
                "Loss Reduction vs Nifty": "{:+.2%}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    stress_results = data[
        "historical_stress_results"
    ]

    display_columns = [
        column
        for column in [
            "Stress Scenario",
            "Portfolio",
            "Start Date",
            "End Date",
            "Trading Days",
            "Cumulative Return",
            "Maximum Drawdown",
            "Worst Daily Return",
            "Loss Reduction vs Nifty",
        ]
        if column in stress_results.columns
    ]

    st.dataframe(
        stress_results[
            display_columns
        ].style.format(
            {
                "Cumulative Return": "{:.2%}",
                "Maximum Drawdown": "{:.2%}",
                "Worst Daily Return": "{:.2%}",
                "Loss Reduction vs Nifty": "{:+.2%}",
            },
            na_rep="—",
        ),
        width="stretch",
        hide_index=True,
    )

    st.info(
        "No portfolio dominated every stress period. "
        "Maximum Sharpe performed best in three selected windows, "
        "while Minimum Volatility performed best in two."
    )


def render_risk_simulation(data):
    """
    Render VaR, Expected Shortfall, Monte Carlo and scenarios.
    """

    st.markdown("## VaR and Expected Shortfall")

    one_day_risk = data[
        "one_day_risk_summary"
    ]

    st.dataframe(
        one_day_risk.style.format(
            {
                "1-Day VaR 95%": "{:.2%}",
                "1-Day ES 95%": "{:.2%}",
                "1-Day VaR 99%": "{:.2%}",
                "1-Day ES 99%": "{:.2%}",
                "Worst Historical Day": "{:.2%}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.markdown("## One-year Monte Carlo simulation")

    monte_carlo = data[
        "monte_carlo_risk_summary"
    ]

    selected_columns = [
        column
        for column in [
            "Median Terminal Return",
            "5th Percentile Return",
            "95th Percentile Return",
            "Probability of Loss",
            "Probability of Loss > 10%",
            "One-Year VaR 95%",
            "One-Year ES 95%",
            "Median Maximum Drawdown",
            "5th Percentile Maximum Drawdown",
            "Median Terminal Value (₹)",
        ]
        if column in monte_carlo.columns
    ]

    st.dataframe(
        monte_carlo[
            selected_columns
        ].style.format(
            {
                "Median Terminal Return": "{:.2%}",
                "5th Percentile Return": "{:.2%}",
                "95th Percentile Return": "{:.2%}",
                "Probability of Loss": "{:.2%}",
                "Probability of Loss > 10%": "{:.2%}",
                "One-Year VaR 95%": "{:.2%}",
                "One-Year ES 95%": "{:.2%}",
                "Median Maximum Drawdown": "{:.2%}",
                "5th Percentile Maximum Drawdown": "{:.2%}",
                "Median Terminal Value (₹)": "₹{:,.0f}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.markdown("## Bull, base and bear scenarios")

    st.dataframe(
        data[
            "scenario_return_comparison"
        ].style.format(
            "{:.2%}",
            na_rep="—",
        ),
        width="stretch",
    )

    st.caption(
        "Monte Carlo and scenario outputs are model-based risk "
        "estimates, not guarantees or investment forecasts."
    )


def render_user_portfolio_preview(data):
    """
    Preview the validated custom-portfolio research case.
    """

    st.markdown("## User Portfolio Lab")

    st.success(
        "The research engine now supports user-selected NSE and "
        "BSE stocks entered through weights, invested amounts "
        "or quantities."
    )

    st.markdown("### Validated six-stock example")

    st.dataframe(
        data[
            "user_portfolio_optimised_weights"
        ].style.format(
            "{:.2%}",
            na_rep="—",
        ),
        width="stretch",
    )

    st.markdown("### Custom-portfolio model comparison")

    custom_model = data[
        "user_portfolio_model_comparison"
    ]

    st.dataframe(
        custom_model.style.format(
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

    st.info(
        "The interactive stock-entry controls and on-demand "
        "calculation engine will be connected in the next step."
    )


def render_optimisation_page():
    """
    Render the v0.5 research dashboard.
    """

    st.title(
        "⚖️ Portfolio Optimisation & Stress Testing"
    )

    st.markdown(
        """
        **Version v0.5 — Optimisation, Stress Testing and
        User Portfolio Lab**

        This module compares current, minimum-volatility,
        maximum-Sharpe and risk-parity portfolios using Indian
        equity data, realistic constraints and out-of-sample tests.
        """
    )

    st.info(
        "This application is an academic financial-analytics "
        "project and does not constitute investment advice."
    )

    try:
        data = load_v05_dashboard_data()

    except Exception as error:
        st.error(
            "The v0.5 datasets could not be loaded."
        )
        st.exception(error)
        st.stop()

    (
        overview_tab,
        optimisation_tab,
        out_of_sample_tab,
        stress_tab,
        risk_tab,
        user_portfolio_tab,
    ) = st.tabs(
        [
            "Research Overview",
            "Optimisation",
            "Out-of-Sample Test",
            "Stress Testing",
            "Monte Carlo & Scenarios",
            "User Portfolio Lab",
        ]
    )

    with overview_tab:
        render_research_overview(data)

    with optimisation_tab:
        render_optimisation(data)

    with out_of_sample_tab:
        render_out_of_sample(data)

    with stress_tab:
        render_stress_testing(data)

    with risk_tab:
        render_risk_simulation(data)

    with user_portfolio_tab:
        render_user_portfolio_preview(data)

    with st.expander(
        "v0.5 dataset manifest"
    ):
        st.dataframe(
            load_optimisation_manifest(),
            width="stretch",
            hide_index=True,
        )


render_optimisation_page()
