
import pandas as pd
import streamlit as st

from src.ui.forecasting_data import (
    build_forecasting_manifest,
    load_forecasting_data,
)


PAGE_TITLE = "Forecasting & Risk Models"


def format_date(value):
    """
    Format a date value for dashboard display.
    """

    parsed_date = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(parsed_date):
        return "Not available"

    return parsed_date.strftime(
        "%d %B %Y"
    )


def render_overview(data):
    """
    Render the main v0.4 research conclusions.
    """

    research_summary = data[
        "v04_research_summary"
    ]

    model_roles = data[
        "final_volatility_model_roles"
    ]

    regime_summary = data[
        "latest_forecast_regime_summary"
    ].iloc[0]

    performance = data[
        "volatility_allocation_comparison"
    ]

    st.markdown("## Research conclusion")

    st.info(
        "The v0.4 research found that daily return direction was "
        "not forecast reliably, while dynamic volatility forecasts "
        "were more useful for risk monitoring."
    )

    strategy_name = (
        "63-Day Historical Volatility"
    )

    nifty_name = "Nifty 50"

    strategy_result = performance.loc[
        strategy_name
    ]

    nifty_result = performance.loc[
        nifty_name
    ]

    metric_1, metric_2, metric_3, metric_4 = (
        st.columns(4)
    )

    metric_1.metric(
        "Preferred allocation CAGR",
        f"{strategy_result['CAGR']:.2%}",
        delta=(
            f"{strategy_result['CAGR'] - nifty_result['CAGR']:+.2%} "
            "vs Nifty"
        ),
    )

    metric_2.metric(
        "Preferred allocation Sharpe",
        f"{strategy_result['Sharpe Ratio']:.2f}",
    )

    metric_3.metric(
        "Maximum drawdown",
        f"{strategy_result['Maximum Drawdown']:.2%}",
    )

    metric_4.metric(
        "Beta vs Nifty",
        f"{strategy_result['Beta vs Nifty']:.2f}",
    )

    st.markdown("### Final model roles")

    st.dataframe(
        model_roles,
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Latest market regime")

    regime_col_1, regime_col_2, regime_col_3 = (
        st.columns(3)
    )

    regime_col_1.metric(
        "Signal date",
        format_date(
            regime_summary["Signal Date"]
        ),
    )

    regime_col_2.metric(
        "Market regime",
        str(
            regime_summary["Market Regime"]
        ),
    )

    regime_col_3.metric(
        "Target exposure",
        (
            f"{regime_summary['Equity Allocation']:.0%} equity / "
            f"{regime_summary['Cash Allocation']:.0%} cash"
        ),
    )

    st.caption(
        "The market regime is determined by comparing the Nifty 50 "
        "with its 200-day moving average."
    )

    st.markdown("### v0.4 research summary")

    st.dataframe(
        research_summary,
        width="stretch",
        hide_index=True,
    )


def render_arima_analysis(data):
    """
    Render ARIMA selection, diagnostics and forecast evaluation.
    """

    order_selection = data[
        "arima_order_selection"
    ]

    diagnostics = data[
        "arima_residual_diagnostics"
    ]

    accuracy = data[
        "arima_forecast_accuracy"
    ]

    interval_summary = data[
        "arima_interval_summary"
    ]

    stationarity = data[
        "stationarity_results"
    ]

    st.markdown("## ARIMA return forecasting")

    st.warning(
        "ARIMA(2, 0, 3) was the preferred in-sample specification, "
        "but it failed to beat the zero-return benchmark during the "
        "out-of-sample period."
    )

    stationary_count = int(
        stationarity[
            "Stationary at 5%"
        ].sum()
    )

    stationarity_col_1, stationarity_col_2 = (
        st.columns(2)
    )

    stationarity_col_1.metric(
        "Series tested",
        f"{len(stationarity):,}",
    )

    stationarity_col_2.metric(
        "Stationary at 5%",
        f"{stationary_count:,}",
    )

    st.markdown("### ARIMA order selection")

    order_display_columns = [
        column
        for column in [
            "Order",
            "AIC",
            "BIC",
            "HQIC",
            "Log Likelihood",
            "Parameters",
        ]
        if column in order_selection.columns
    ]

    st.dataframe(
        order_selection[
            order_display_columns
        ].head(10),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Out-of-sample forecast accuracy")

    st.dataframe(
        accuracy.style.format(
            {
                "Observations": "{:,.0f}",
                "MAE": "{:.4%}",
                "RMSE": "{:.4%}",
                "Forecast Bias": "{:+.4%}",
                "Directional Accuracy": "{:.2%}",
                "Forecast–Actual Correlation": "{:.4f}",
                "RMSE Improvement vs Zero": "{:+.2%}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.markdown("### Residual diagnostics")

    st.dataframe(
        diagnostics.style.format(
            {
                "Test Statistic": "{:.4f}",
                "P-Value": "{:.6f}",
            }
        ),
        width="stretch",
    )

    st.markdown("### Forecast interval")

    interval_display = (
        interval_summary
        .copy()
    )

    if "Result" in interval_display.columns:

        interval_display["Result"] = (
            interval_display["Result"]
            .map(
                lambda value: (
                    f"{value:.2%}"
                    if pd.notna(value)
                    else "—"
                )
            )
        )

    st.dataframe(
        interval_display,
        width="stretch",
    )


def render_volatility_analysis(data):
    """
    Render volatility-model selection and evaluation.
    """

    model_selection = data[
        "volatility_model_selection"
    ]

    forecast_accuracy = data[
        "volatility_forecast_accuracy"
    ]

    diagnostics = data[
        "garch_residual_diagnostics"
    ]

    latest_forecasts = data[
        "latest_stock_volatility_forecasts"
    ]

    st.markdown("## Dynamic volatility forecasting")

    st.success(
        "GJR-GARCH(1,1) with Student's-t innovations successfully "
        "captured the principal volatility clustering and remains the "
        "primary dynamic risk-monitoring model."
    )

    st.markdown("### Candidate-model comparison")

    st.dataframe(
        model_selection.style.format(
            {
                "AIC": "{:.2f}",
                "BIC": "{:.2f}",
                "Log Likelihood": "{:.2f}",
                "Parameters": "{:.0f}",
                "Alpha": "{:.4f}",
                "Beta": "{:.4f}",
                "Gamma": "{:.4f}",
                "Persistence": "{:.4f}",
                "Degrees of Freedom": "{:.2f}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.markdown("### Out-of-sample volatility accuracy")

    st.dataframe(
        forecast_accuracy.style.format(
            {
                "Observations": "{:,.0f}",
                "Volatility MAE": "{:.4f}",
                "Volatility RMSE": "{:.4f}",
                "Variance MSE": "{:.4f}",
                "QLIKE": "{:.4f}",
                "Volatility–Absolute Residual Correlation":
                    "{:.4f}",
                "Average Forecast Volatility": "{:.4f}",
                "QLIKE Improvement vs 63-Day": "{:+.4f}",
            }
        ),
        width="stretch",
    )

    st.markdown("### Standardised-residual diagnostics")

    st.dataframe(
        diagnostics.style.format(
            {
                "Test Statistic": "{:.4f}",
                "P-Value": "{:.6f}",
            }
        ),
        width="stretch",
    )

    st.markdown("### Latest India 10 volatility forecasts")

    st.dataframe(
        latest_forecasts.style.format(
            {
                "Observations": "{:,.0f}",
                "Daily Volatility Forecast (%)": "{:.3f}%",
                "Annualised Volatility Forecast (%)": "{:.2f}%",
                "Alpha": "{:.4f}",
                "Beta": "{:.4f}",
                "Gamma": "{:.4f}",
                "Persistence": "{:.4f}",
                "Degrees of Freedom": "{:.2f}",
            },
            na_rep="—",
        ),
        width="stretch",
    )


def render_allocation_analysis(data):
    """
    Render the latest forecast-aware portfolio allocation.
    """

    allocation = data[
        "latest_forecast_aware_allocation"
    ]

    regime_summary = data[
        "latest_forecast_regime_summary"
    ].iloc[0]

    st.markdown("## Latest forecast-aware allocation")

    allocation_col_1, allocation_col_2, allocation_col_3 = (
        st.columns(3)
    )

    allocation_col_1.metric(
        "Market regime",
        str(
            regime_summary["Market Regime"]
        ),
    )

    allocation_col_2.metric(
        "Equity allocation",
        f"{regime_summary['Equity Allocation']:.2%}",
    )

    allocation_col_3.metric(
        "Cash allocation",
        f"{regime_summary['Cash Allocation']:.2%}",
    )

    st.dataframe(
        allocation.style.format(
            {
                "Forecast Annualised Volatility (%)": "{:.2f}%",
                "Six-Month Momentum": "{:+.2%}",
                "Momentum Score": "{:.3f}",
                "Low-Volatility Score": "{:.3f}",
                "Growth Composite Score": "{:.3f}",
                "Equity-Only Weight": "{:.2%}",
                "Final Portfolio Weight": "{:.2%}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.info(
        "This allocation is presented as a research output. The final "
        "model decision retains 63-day historical volatility as the "
        "default allocation input because it performed better in the "
        "like-for-like backtest."
    )


def render_backtest_analysis(data):
    """
    Render the portfolio comparison and trading analysis.
    """

    performance = data[
        "volatility_allocation_comparison"
    ]

    trading = data[
        "volatility_model_trading_comparison"
    ]

    forecast_daily = data[
        "forecast_aware_backtest_daily"
    ]

    historical_daily = data[
        "historical_volatility_backtest_daily"
    ]

    st.markdown("## Allocation-model backtest")

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
                "Beta vs Nifty": "{:.2f}",
                "Correlation vs Nifty": "{:.2f}",
            }
        ),
        width="stretch",
    )

    chart_data = pd.DataFrame(
        {
            "Date":
                forecast_daily["Date"],

            "GJR-GARCH Forecast-Aware":
                forecast_daily[
                    "Portfolio Value (₹)"
                ],

            "63-Day Historical Volatility":
                historical_daily[
                    "Portfolio Value (₹)"
                ],
        }
    ).dropna()

    chart_data = (
        chart_data
        .set_index("Date")
    )

    st.markdown("### Growth of ₹10,00,000")

    st.line_chart(
        chart_data,
        width="stretch",
    )

    st.markdown("### Trading comparison")

    st.dataframe(
        trading.style.format(
            {
                "Rebalances": "{:,.0f}",
                "Total One-Way Turnover": "{:.2%}",
                "Average Turnover per Rebalance": "{:.2%}",
                "Annualised One-Way Turnover": "{:.2%}",
                "Total Transaction Costs (₹)": "₹{:,.2f}",
                "Costs as Percentage of Initial Capital": "{:.2%}",
            }
        ),
        width="stretch",
    )

    st.warning(
        "GJR-GARCH produced greater turnover and transaction costs "
        "without improving realised portfolio performance relative "
        "to the simpler 63-day historical-volatility allocation."
    )


def render_forecasting_page():
    """
    Render the complete v0.4 forecasting and risk page.
    """

    st.title("📈 Forecasting & Risk Models")

    st.markdown(
        """
        **Version v0.4 — Forecasting and Risk Models**

        This page evaluates ARIMA return forecasts, ARCH/GARCH risk
        forecasts and the investment usefulness of dynamic volatility
        inside the India 10 Adaptive Barbell portfolio.
        """
    )

    st.info(
        "This application is an academic financial-analytics project "
        "and does not constitute investment advice."
    )

    try:

        forecasting_data = (
            load_forecasting_data()
        )

    except Exception as error:

        st.error(
            "The forecasting datasets could not be loaded."
        )

        st.exception(error)
        st.stop()

    (
        overview_tab,
        arima_tab,
        volatility_tab,
        allocation_tab,
        backtest_tab,
    ) = st.tabs(
        [
            "Research Overview",
            "ARIMA Forecasting",
            "Volatility Models",
            "Latest Allocation",
            "Allocation Backtest",
        ]
    )

    with overview_tab:
        render_overview(
            forecasting_data
        )

    with arima_tab:
        render_arima_analysis(
            forecasting_data
        )

    with volatility_tab:
        render_volatility_analysis(
            forecasting_data
        )

    with allocation_tab:
        render_allocation_analysis(
            forecasting_data
        )

    with backtest_tab:
        render_backtest_analysis(
            forecasting_data
        )

    with st.expander(
        "Dataset manifest"
    ):

        st.dataframe(
            build_forecasting_manifest(
                forecasting_data
            ),
            width="stretch",
            hide_index=True,
        )


render_forecasting_page()
