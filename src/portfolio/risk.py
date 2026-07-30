"""
Production portfolio-risk engine for Indian listed equities.

Modules
-------
Concentration and diversification
Historical stress testing
Historical VaR and Expected Shortfall
Block-bootstrap Monte Carlo simulation
Bull, base and bear scenario analysis
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.portfolio.optimisation import (
    TRADING_DAYS_PER_YEAR,
    calculate_portfolio_volatility,
)


DEFAULT_SCENARIO_ASSUMPTIONS = {
    "Bear": {
        "Benchmark Return": -0.20,
        "Volatility Multiplier": 1.50,
    },
    "Base": {
        "Benchmark Return": 0.08,
        "Volatility Multiplier": 1.00,
    },
    "Bull": {
        "Benchmark Return": 0.20,
        "Volatility Multiplier": 0.85,
    },
}


def validate_risk_inputs(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    portfolio_weights: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    """
    Align and validate portfolio-risk inputs.
    """

    if not isinstance(
        asset_returns,
        pd.DataFrame,
    ):
        raise TypeError(
            "Asset returns must be a pandas DataFrame."
        )

    if not isinstance(
        portfolio_weights,
        pd.DataFrame,
    ):
        raise TypeError(
            "Portfolio weights must be a pandas DataFrame."
        )

    clean_asset_returns = (
        asset_returns
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .astype(float)
        .copy()
    )

    clean_benchmark_returns = (
        pd.Series(
            benchmark_returns,
            dtype=float,
            name="Benchmark",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    aligned_data = pd.concat(
        [
            clean_asset_returns,
            clean_benchmark_returns,
        ],
        axis=1,
    ).dropna()

    if len(aligned_data) < 60:
        raise ValueError(
            "At least 60 aligned return observations are required."
        )

    clean_asset_returns = (
        aligned_data[
            clean_asset_returns.columns
        ]
    )

    clean_benchmark_returns = (
        aligned_data[
            "Benchmark"
        ]
    )

    clean_weights = (
        portfolio_weights
        .reindex(
            index=clean_asset_returns.columns
        )
        .astype(float)
    )

    if clean_weights.isna().any().any():
        raise ValueError(
            "Portfolio weights do not match the asset-return columns."
        )

    if (
        clean_weights
        < -1e-10
    ).any().any():
        raise ValueError(
            "Portfolio weights cannot be negative."
        )

    if not np.allclose(
        clean_weights.sum(
            axis=0
        ),
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Every portfolio must sum to 100%."
        )

    return (
        clean_asset_returns,
        clean_benchmark_returns,
        clean_weights,
    )


def calculate_diversification_statistics(
    weights: pd.Series,
    annualised_covariance: pd.DataFrame,
) -> pd.Series:
    """
    Calculate concentration and diversification statistics.
    """

    aligned_weights = (
        pd.Series(
            weights,
            dtype=float,
        )
        .reindex(
            annualised_covariance.index
        )
        .fillna(0.0)
    )

    if aligned_weights.sum() <= 0:
        raise ValueError(
            "Portfolio weights must have a positive total."
        )

    aligned_weights = (
        aligned_weights
        / aligned_weights.sum()
    )

    covariance = (
        annualised_covariance
        .reindex(
            index=aligned_weights.index,
            columns=aligned_weights.index,
        )
        .astype(float)
    )

    asset_volatility = pd.Series(
        np.sqrt(
            np.diag(
                covariance.to_numpy()
            )
        ),
        index=aligned_weights.index,
    )

    portfolio_volatility = (
        calculate_portfolio_volatility(
            aligned_weights,
            covariance,
        )
    )

    weighted_asset_volatility = float(
        aligned_weights
        @ asset_volatility
    )

    diversification_ratio = (
        weighted_asset_volatility
        / portfolio_volatility
        if portfolio_volatility > 0
        else np.nan
    )

    denominator = np.outer(
        asset_volatility,
        asset_volatility,
    )

    correlation_matrix = (
        covariance.to_numpy()
        / denominator
    )

    np.fill_diagonal(
        correlation_matrix,
        np.nan,
    )

    average_pairwise_correlation = (
        float(
            np.nanmean(
                correlation_matrix
            )
        )
    )

    herfindahl_index = float(
        np.square(
            aligned_weights
        ).sum()
    )

    effective_holdings = (
        1.0
        / herfindahl_index
        if herfindahl_index > 0
        else np.nan
    )

    active_holdings = int(
        (
            aligned_weights
            > 0.001
        ).sum()
    )

    return pd.Series(
        {
            "Number of Holdings":
                len(
                    aligned_weights
                ),

            "Active Holdings":
                active_holdings,

            "Largest Holding":
                aligned_weights.max(),

            "Top-3 Concentration":
                aligned_weights.nlargest(
                    3
                ).sum(),

            "Herfindahl Index":
                herfindahl_index,

            "Effective Holdings":
                effective_holdings,

            "Average Pairwise Correlation":
                average_pairwise_correlation,

            "Diversification Ratio":
                diversification_ratio,
        }
    )


def calculate_var_and_expected_shortfall(
    return_array: np.ndarray | pd.Series,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Return positive loss values for historical VaR and ES.
    """

    if not (
        0.0
        < confidence_level
        < 1.0
    ):
        raise ValueError(
            "Confidence level must be between zero and one."
        )

    clean_returns = np.asarray(
        return_array,
        dtype=float,
    )

    clean_returns = clean_returns[
        np.isfinite(
            clean_returns
        )
    ]

    if len(clean_returns) < 20:
        raise ValueError(
            "At least 20 valid returns are required."
        )

    tail_probability = (
        1.0
        - confidence_level
    )

    return_quantile = float(
        np.quantile(
            clean_returns,
            tail_probability,
        )
    )

    tail_returns = clean_returns[
        clean_returns
        <= return_quantile
    ]

    value_at_risk = max(
        0.0,
        -return_quantile,
    )

    expected_shortfall = max(
        0.0,
        -float(
            tail_returns.mean()
        ),
    )

    return (
        value_at_risk,
        expected_shortfall,
    )


def identify_historical_stress_periods(
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """
    Identify five data-driven benchmark stress periods.
    """

    benchmark = (
        pd.Series(
            benchmark_returns,
            dtype=float,
        )
        .dropna()
        .sort_index()
    )

    if len(benchmark) < 126:
        raise ValueError(
            "At least 126 benchmark observations are required."
        )

    benchmark_wealth = (
        1.0
        + benchmark
    ).cumprod()

    running_peak = (
        benchmark_wealth.cummax()
    )

    drawdown = (
        benchmark_wealth
        / running_peak
        - 1.0
    )

    drawdown_trough_date = (
        drawdown.idxmin()
    )

    drawdown_peak_date = (
        benchmark_wealth.loc[
            :drawdown_trough_date
        ]
        .idxmax()
    )

    worst_single_day = (
        benchmark.idxmin()
    )

    rolling_21_return = (
        1.0
        + benchmark
    ).rolling(
        21
    ).apply(
        np.prod,
        raw=True,
    ) - 1.0

    rolling_63_return = (
        1.0
        + benchmark
    ).rolling(
        63
    ).apply(
        np.prod,
        raw=True,
    ) - 1.0

    rolling_63_volatility = (
        benchmark.rolling(
            63
        ).std(
            ddof=1
        )
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )

    worst_21_end = (
        rolling_21_return.idxmin()
    )

    worst_63_end = (
        rolling_63_return.idxmin()
    )

    highest_volatility_end = (
        rolling_63_volatility.idxmax()
    )

    benchmark_index = (
        benchmark.index
    )

    def window_start(
        end_date,
        window_length,
    ):
        end_position = (
            benchmark_index.get_loc(
                end_date
            )
        )

        start_position = max(
            0,
            end_position
            - window_length
            + 1,
        )

        return benchmark_index[
            start_position
        ]

    stress_periods = pd.DataFrame(
        [
            {
                "Stress Scenario":
                    "Worst Benchmark Single Day",

                "Start Date":
                    worst_single_day,

                "End Date":
                    worst_single_day,

                "Selection Basis":
                    "Lowest daily benchmark return",
            },
            {
                "Stress Scenario":
                    "Worst Benchmark 21-Day Window",

                "Start Date":
                    window_start(
                        worst_21_end,
                        21,
                    ),

                "End Date":
                    worst_21_end,

                "Selection Basis":
                    "Lowest compounded 21-day benchmark return",
            },
            {
                "Stress Scenario":
                    "Worst Benchmark 63-Day Window",

                "Start Date":
                    window_start(
                        worst_63_end,
                        63,
                    ),

                "End Date":
                    worst_63_end,

                "Selection Basis":
                    "Lowest compounded 63-day benchmark return",
            },
            {
                "Stress Scenario":
                    "Highest Benchmark 63-Day Volatility",

                "Start Date":
                    window_start(
                        highest_volatility_end,
                        63,
                    ),

                "End Date":
                    highest_volatility_end,

                "Selection Basis":
                    "Highest annualised 63-day benchmark volatility",
            },
            {
                "Stress Scenario":
                    "Maximum Benchmark Drawdown",

                "Start Date":
                    drawdown_peak_date,

                "End Date":
                    drawdown_trough_date,

                "Selection Basis":
                    "Largest benchmark peak-to-trough decline",
            },
        ]
    )

    return stress_periods


def run_historical_stress_test(
    portfolio_daily_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> dict[str, pd.DataFrame]:
    """
    Test portfolios during five data-driven benchmark stress periods.
    """

    portfolio_returns = (
        portfolio_daily_returns
        .astype(float)
        .copy()
    )

    benchmark = (
        pd.Series(
            benchmark_returns,
            dtype=float,
            name="Benchmark",
        )
    )

    aligned_returns = pd.concat(
        [
            portfolio_returns,
            benchmark,
        ],
        axis=1,
    ).dropna()

    stress_periods = (
        identify_historical_stress_periods(
            aligned_returns[
                "Benchmark"
            ]
        )
    )

    stress_records = []

    for _, scenario in (
        stress_periods.iterrows()
    ):

        scenario_name = (
            scenario[
                "Stress Scenario"
            ]
        )

        start_date = pd.Timestamp(
            scenario[
                "Start Date"
            ]
        )

        end_date = pd.Timestamp(
            scenario[
                "End Date"
            ]
        )

        period_returns = aligned_returns.loc[
            start_date:
            end_date
        ]

        benchmark_period_returns = (
            period_returns[
                "Benchmark"
            ]
        )

        benchmark_cumulative_return = float(
            (
                1.0
                + benchmark_period_returns
            ).prod()
            - 1.0
        )

        for portfolio_name in (
            aligned_returns.columns
        ):

            selected_returns = (
                period_returns[
                    portfolio_name
                ]
            )

            wealth = (
                1.0
                + selected_returns
            ).cumprod()

            drawdown = (
                wealth
                / wealth.cummax()
                - 1.0
            )

            cumulative_return = float(
                wealth.iloc[-1]
                - 1.0
            )

            annualised_volatility = (
                selected_returns.std(
                    ddof=1
                )
                * np.sqrt(
                    TRADING_DAYS_PER_YEAR
                )
                if len(
                    selected_returns
                ) > 1
                else 0.0
            )

            loss_reduction = (
                cumulative_return
                - benchmark_cumulative_return
            )

            stress_records.append(
                {
                    "Stress Scenario":
                        scenario_name,

                    "Portfolio":
                        portfolio_name,

                    "Start Date":
                        start_date,

                    "End Date":
                        end_date,

                    "Trading Days":
                        len(
                            selected_returns
                        ),

                    "Cumulative Return":
                        cumulative_return,

                    "Annualised Volatility":
                        annualised_volatility,

                    "Maximum Drawdown":
                        float(
                            drawdown.min()
                        ),

                    "Worst Daily Return":
                        float(
                            selected_returns.min()
                        ),

                    "Benchmark Return":
                        benchmark_cumulative_return,

                    "Loss Reduction vs Benchmark":
                        loss_reduction,
                }
            )

    stress_results = pd.DataFrame(
        stress_records
    )

    portfolio_only_results = (
        stress_results.loc[
            stress_results[
                "Portfolio"
            ]
            != "Benchmark"
        ]
    )

    winner_indices = (
        portfolio_only_results
        .groupby(
            "Stress Scenario"
        )[
            "Cumulative Return"
        ]
        .idxmax()
    )

    winners = (
        portfolio_only_results.loc[
            winner_indices
        ][
            [
                "Stress Scenario",
                "Portfolio",
                "Cumulative Return",
                "Benchmark Return",
                "Loss Reduction vs Benchmark",
            ]
        ]
        .rename(
            columns={
                "Portfolio":
                    "Best Portfolio",

                "Cumulative Return":
                    "Best Portfolio Return",
            }
        )
        .set_index(
            "Stress Scenario"
        )
    )

    return {
        "periods":
            stress_periods,

        "results":
            stress_results,

        "winners":
            winners,
    }


def run_monte_carlo_risk_analysis(
    asset_returns: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    expected_returns: pd.Series,
    initial_value_inr: float = 1_000_000,
    simulations: int = 10_000,
    horizon_days: int = 252,
    block_length: int = 21,
    estimation_window: int = 756,
    random_seed: int = 42,
    chunk_size: int = 500,
) -> dict[str, Any]:
    """
    Run cross-sectional historical block-bootstrap simulations.
    """

    if simulations < 100:
        raise ValueError(
            "At least 100 simulations are required."
        )

    if horizon_days < 1:
        raise ValueError(
            "Simulation horizon must be positive."
        )

    if block_length < 1:
        raise ValueError(
            "Bootstrap block length must be positive."
        )

    estimation_returns = (
        asset_returns
        .tail(
            estimation_window
        )
        .copy()
    )

    if len(estimation_returns) < max(
        60,
        block_length
        + 1,
    ):
        raise ValueError(
            "The estimation period is too short."
        )

    aligned_expected_returns = (
        pd.Series(
            expected_returns,
            dtype=float,
        )
        .reindex(
            estimation_returns.columns
        )
    )

    if aligned_expected_returns.isna().any():
        raise ValueError(
            "Expected returns do not match the assets."
        )

    aligned_weights = (
        portfolio_weights
        .reindex(
            index=estimation_returns.columns
        )
    )

    if aligned_weights.isna().any().any():
        raise ValueError(
            "Portfolio weights do not match the assets."
        )

    target_daily_returns = (
        aligned_expected_returns
        / TRADING_DAYS_PER_YEAR
    )

    adjusted_bootstrap_returns = (
        estimation_returns
        - estimation_returns.mean()
        + target_daily_returns
    )

    adjusted_return_array = (
        adjusted_bootstrap_returns
        .to_numpy(
            dtype=float
        )
    )

    weight_matrix = (
        aligned_weights
        .to_numpy(
            dtype=float
        )
    )

    portfolio_names = (
        aligned_weights.columns.tolist()
    )

    number_of_blocks = int(
        np.ceil(
            horizon_days
            / block_length
        )
    )

    maximum_block_start = (
        len(
            adjusted_return_array
        )
        - block_length
    )

    random_generator = (
        np.random.default_rng(
            random_seed
        )
    )

    terminal_return_chunks = []
    maximum_drawdown_chunks = []

    completed_simulations = 0

    while completed_simulations < simulations:

        current_chunk_size = min(
            chunk_size,
            simulations
            - completed_simulations,
        )

        block_start_indices = (
            random_generator.integers(
                low=0,
                high=maximum_block_start + 1,
                size=(
                    current_chunk_size,
                    number_of_blocks,
                ),
            )
        )

        block_offsets = np.arange(
            block_length
        )

        sampled_indices = (
            block_start_indices[
                :,
                :,
                None,
            ]
            + block_offsets[
                None,
                None,
                :,
            ]
        )

        sampled_indices = (
            sampled_indices
            .reshape(
                current_chunk_size,
                -1,
            )
            [
                :,
                :horizon_days,
            ]
        )

        simulated_asset_returns = (
            adjusted_return_array[
                sampled_indices
            ]
        )

        simulated_asset_wealth = np.cumprod(
            1.0
            + simulated_asset_returns,
            axis=1,
        )

        simulated_portfolio_wealth = np.einsum(
            "sda,ap->sdp",
            simulated_asset_wealth,
            weight_matrix,
        )

        terminal_returns = (
            simulated_portfolio_wealth[
                :,
                -1,
                :,
            ]
            - 1.0
        )

        running_peaks = np.maximum.accumulate(
            simulated_portfolio_wealth,
            axis=1,
        )

        simulated_drawdowns = (
            simulated_portfolio_wealth
            / running_peaks
            - 1.0
        )

        maximum_drawdowns = (
            simulated_drawdowns.min(
                axis=1
            )
        )

        terminal_return_chunks.append(
            terminal_returns
        )

        maximum_drawdown_chunks.append(
            maximum_drawdowns
        )

        completed_simulations += (
            current_chunk_size
        )

    simulated_terminal_returns = np.vstack(
        terminal_return_chunks
    )

    simulated_maximum_drawdowns = np.vstack(
        maximum_drawdown_chunks
    )

    summary_records = []

    for portfolio_number, portfolio_name in enumerate(
        portfolio_names
    ):

        terminal_returns = (
            simulated_terminal_returns[
                :,
                portfolio_number,
            ]
        )

        drawdowns = (
            simulated_maximum_drawdowns[
                :,
                portfolio_number,
            ]
        )

        (
            one_year_var_95,
            one_year_es_95,
        ) = calculate_var_and_expected_shortfall(
            terminal_returns,
            confidence_level=0.95,
        )

        (
            one_year_var_99,
            one_year_es_99,
        ) = calculate_var_and_expected_shortfall(
            terminal_returns,
            confidence_level=0.99,
        )

        summary_records.append(
            {
                "Portfolio":
                    portfolio_name,

                "Mean Terminal Return":
                    terminal_returns.mean(),

                "Median Terminal Return":
                    np.median(
                        terminal_returns
                    ),

                "5th Percentile Return":
                    np.quantile(
                        terminal_returns,
                        0.05,
                    ),

                "95th Percentile Return":
                    np.quantile(
                        terminal_returns,
                        0.95,
                    ),

                "Probability of Loss":
                    np.mean(
                        terminal_returns < 0
                    ),

                "Probability of Loss > 10%":
                    np.mean(
                        terminal_returns < -0.10
                    ),

                "Probability of Loss > 20%":
                    np.mean(
                        terminal_returns < -0.20
                    ),

                "One-Year VaR 95%":
                    one_year_var_95,

                "One-Year ES 95%":
                    one_year_es_95,

                "One-Year VaR 99%":
                    one_year_var_99,

                "One-Year ES 99%":
                    one_year_es_99,

                "Median Maximum Drawdown":
                    np.median(
                        drawdowns
                    ),

                "5th Percentile Maximum Drawdown":
                    np.quantile(
                        drawdowns,
                        0.05,
                    ),

                "Median Terminal Value (₹)":
                    initial_value_inr
                    * (
                        1.0
                        + np.median(
                            terminal_returns
                        )
                    ),

                "5th Percentile Terminal Value (₹)":
                    initial_value_inr
                    * (
                        1.0
                        + np.quantile(
                            terminal_returns,
                            0.05,
                        )
                    ),
            }
        )

    summary = (
        pd.DataFrame(
            summary_records
        )
        .set_index(
            "Portfolio"
        )
    )

    return {
        "summary":
            summary,

        "terminal_returns":
            simulated_terminal_returns,

        "maximum_drawdowns":
            simulated_maximum_drawdowns,

        "portfolio_names":
            portfolio_names,
    }


def run_scenario_analysis(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    portfolio_weights: pd.DataFrame,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    initial_value_inr: float = 1_000_000,
    scenario_assumptions: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run beta-adjusted bull, base and bear scenarios.
    """

    assumptions = (
        scenario_assumptions
        if scenario_assumptions is not None
        else DEFAULT_SCENARIO_ASSUMPTIONS
    )

    required_scenarios = {
        "Bear",
        "Base",
        "Bull",
    }

    if set(
        assumptions
    ) != required_scenarios:
        raise ValueError(
            "Scenario assumptions must contain Bear, Base and Bull."
        )

    aligned_data = pd.concat(
        [
            asset_returns,
            benchmark_returns.rename(
                "Benchmark"
            ),
        ],
        axis=1,
    ).dropna()

    benchmark = (
        aligned_data[
            "Benchmark"
        ]
    )

    benchmark_variance = (
        benchmark.var(
            ddof=1
        )
    )

    if benchmark_variance <= 0:
        raise ValueError(
            "Benchmark variance must be greater than zero."
        )

    stock_betas = {}

    for ticker in (
        asset_returns.columns
    ):

        stock_betas[
            ticker
        ] = (
            aligned_data[
                ticker
            ]
            .cov(
                benchmark
            )
            / benchmark_variance
        )

    stock_beta_series = pd.Series(
        stock_betas
    )

    aligned_expected_returns = (
        expected_returns.reindex(
            asset_returns.columns
        )
    )

    base_market_return = (
        assumptions[
            "Base"
        ][
            "Benchmark Return"
        ]
    )

    scenario_records = []

    for scenario_name, scenario_values in (
        assumptions.items()
    ):

        market_return_shock = (
            scenario_values[
                "Benchmark Return"
            ]
            - base_market_return
        )

        scenario_asset_returns = (
            aligned_expected_returns
            + stock_beta_series
            * market_return_shock
        ).clip(
            lower=-0.60,
            upper=0.80,
        )

        for portfolio_name in (
            portfolio_weights.columns
        ):

            weights = (
                portfolio_weights[
                    portfolio_name
                ]
            )

            portfolio_return = float(
                weights
                @ scenario_asset_returns
            )

            base_volatility = (
                calculate_portfolio_volatility(
                    weights,
                    covariance_matrix,
                )
            )

            stressed_volatility = (
                base_volatility
                * scenario_values[
                    "Volatility Multiplier"
                ]
            )

            scenario_records.append(
                {
                    "Scenario":
                        scenario_name,

                    "Portfolio":
                        portfolio_name,

                    "Benchmark Return":
                        scenario_values[
                            "Benchmark Return"
                        ],

                    "Portfolio Return":
                        portfolio_return,

                    "Return Difference vs Benchmark":
                        (
                            portfolio_return
                            - scenario_values[
                                "Benchmark Return"
                            ]
                        ),

                    "Base Volatility":
                        base_volatility,

                    "Stressed Volatility":
                        stressed_volatility,

                    "Ending Value (₹)":
                        initial_value_inr
                        * (
                            1.0
                            + portfolio_return
                        ),

                    "Gain/Loss (₹)":
                        initial_value_inr
                        * portfolio_return,
                }
            )

    scenario_results = pd.DataFrame(
        scenario_records
    )

    scenario_return_table = (
        scenario_results
        .pivot(
            index="Portfolio",
            columns="Scenario",
            values="Portfolio Return",
        )
        .reindex(
            columns=[
                "Bear",
                "Base",
                "Bull",
            ]
        )
    )

    scenario_volatility_table = (
        scenario_results
        .pivot(
            index="Portfolio",
            columns="Scenario",
            values="Stressed Volatility",
        )
        .reindex(
            columns=[
                "Bear",
                "Base",
                "Bull",
            ]
        )
    )

    return {
        "results":
            scenario_results,

        "return_table":
            scenario_return_table,

        "volatility_table":
            scenario_volatility_table,

        "assumptions":
            pd.DataFrame(
                assumptions
            ).T,
    }


def analyse_user_portfolio_risk(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    portfolio_weights: pd.DataFrame,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    initial_value_inr: float = 1_000_000,
    simulations: int = 10_000,
    horizon_days: int = 252,
    block_length: int = 21,
    estimation_window: int = 756,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Run the complete production risk workflow.
    """

    (
        clean_asset_returns,
        clean_benchmark_returns,
        clean_weights,
    ) = validate_risk_inputs(
        asset_returns=asset_returns,
        benchmark_returns=benchmark_returns,
        portfolio_weights=portfolio_weights,
    )

    diversification_records = {}

    for portfolio_name in (
        clean_weights.columns
    ):

        diversification_records[
            portfolio_name
        ] = (
            calculate_diversification_statistics(
                weights=clean_weights[
                    portfolio_name
                ],
                annualised_covariance=(
                    covariance_matrix
                ),
            )
        )

    diversification = (
        pd.DataFrame(
            diversification_records
        )
        .T
    )

    daily_portfolio_returns = (
        clean_asset_returns
        @ clean_weights
    )

    one_day_risk_records = []

    for portfolio_name in (
        daily_portfolio_returns.columns
    ):

        return_array = (
            daily_portfolio_returns[
                portfolio_name
            ]
            .to_numpy()
        )

        (
            var_95,
            es_95,
        ) = calculate_var_and_expected_shortfall(
            return_array,
            confidence_level=0.95,
        )

        (
            var_99,
            es_99,
        ) = calculate_var_and_expected_shortfall(
            return_array,
            confidence_level=0.99,
        )

        one_day_risk_records.append(
            {
                "Portfolio":
                    portfolio_name,

                "1-Day VaR 95%":
                    var_95,

                "1-Day ES 95%":
                    es_95,

                "1-Day VaR 99%":
                    var_99,

                "1-Day ES 99%":
                    es_99,

                "Worst Historical Day":
                    return_array.min(),

                "95% VaR Amount (₹)":
                    initial_value_inr
                    * var_95,

                "95% ES Amount (₹)":
                    initial_value_inr
                    * es_95,
            }
        )

    one_day_risk = (
        pd.DataFrame(
            one_day_risk_records
        )
        .set_index(
            "Portfolio"
        )
    )

    stress = run_historical_stress_test(
        portfolio_daily_returns=(
            daily_portfolio_returns
        ),
        benchmark_returns=(
            clean_benchmark_returns
        ),
    )

    monte_carlo = (
        run_monte_carlo_risk_analysis(
            asset_returns=(
                clean_asset_returns
            ),
            portfolio_weights=(
                clean_weights
            ),
            expected_returns=(
                expected_returns
            ),
            initial_value_inr=(
                initial_value_inr
            ),
            simulations=(
                simulations
            ),
            horizon_days=(
                horizon_days
            ),
            block_length=(
                block_length
            ),
            estimation_window=(
                estimation_window
            ),
            random_seed=(
                random_seed
            ),
        )
    )

    scenarios = run_scenario_analysis(
        asset_returns=(
            clean_asset_returns
        ),
        benchmark_returns=(
            clean_benchmark_returns
        ),
        portfolio_weights=(
            clean_weights
        ),
        expected_returns=(
            expected_returns
        ),
        covariance_matrix=(
            covariance_matrix
        ),
        initial_value_inr=(
            initial_value_inr
        ),
    )

    return {
        "daily_returns":
            daily_portfolio_returns,

        "diversification":
            diversification,

        "one_day_risk":
            one_day_risk,

        "stress_periods":
            stress[
                "periods"
            ],

        "stress_results":
            stress[
                "results"
            ],

        "stress_winners":
            stress[
                "winners"
            ],

        "monte_carlo":
            monte_carlo[
                "summary"
            ],

        "simulation_terminal_returns":
            monte_carlo[
                "terminal_returns"
            ],

        "simulation_drawdowns":
            monte_carlo[
                "maximum_drawdowns"
            ],

        "scenario_results":
            scenarios[
                "results"
            ],

        "scenario_return_table":
            scenarios[
                "return_table"
            ],

        "scenario_volatility_table":
            scenarios[
                "volatility_table"
            ],

        "scenario_assumptions":
            scenarios[
                "assumptions"
            ],
    }
