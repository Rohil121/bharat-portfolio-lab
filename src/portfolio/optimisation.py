"""
Production portfolio-optimisation engine for Indian equities.

Supported portfolio methods:
    Current portfolio
    Minimum volatility
    Maximum Sharpe ratio
    Equal risk contribution / risk parity
    Constrained efficient frontier
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.065


def clean_return_data(
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and clean aligned daily asset returns.
    """

    if not isinstance(
        asset_returns,
        pd.DataFrame,
    ):
        raise TypeError(
            "Asset returns must be a pandas DataFrame."
        )

    clean_returns = (
        asset_returns
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .astype(float)
        .copy()
    )

    if clean_returns.shape[1] < 2:
        raise ValueError(
            "At least two stocks are required."
        )

    if len(clean_returns) < 60:
        raise ValueError(
            "At least 60 aligned return observations are required."
        )

    if clean_returns.columns.duplicated().any():
        raise ValueError(
            "Asset-return columns must be unique."
        )

    return clean_returns


def estimate_portfolio_inputs(
    asset_returns: pd.DataFrame,
    expected_return_shrinkage: float = 0.50,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Estimate shrunk expected returns and Ledoit-Wolf covariance.

    Historical arithmetic returns are shrunk toward the
    cross-sectional average to reduce extreme estimates.
    """

    clean_returns = clean_return_data(
        asset_returns
    )

    if not (
        0.0
        <= expected_return_shrinkage
        <= 1.0
    ):
        raise ValueError(
            "Expected-return shrinkage must be between 0 and 1."
        )

    historical_expected_returns = (
        clean_returns.mean()
        * TRADING_DAYS_PER_YEAR
    )

    cross_sectional_average = (
        historical_expected_returns.mean()
    )

    shrunk_expected_returns = (
        (
            1.0
            - expected_return_shrinkage
        )
        * historical_expected_returns
        + expected_return_shrinkage
        * cross_sectional_average
    )

    covariance_estimator = (
        LedoitWolf()
        .fit(
            clean_returns.to_numpy()
        )
    )

    annualised_covariance = (
        pd.DataFrame(
            covariance_estimator.covariance_,
            index=clean_returns.columns,
            columns=clean_returns.columns,
        )
        * TRADING_DAYS_PER_YEAR
    )

    return (
        shrunk_expected_returns,
        annualised_covariance,
    )


def calculate_portfolio_return(
    weights: pd.Series | np.ndarray,
    expected_returns: pd.Series | np.ndarray,
) -> float:
    """
    Calculate model-implied annual portfolio return.
    """

    return float(
        np.asarray(
            weights,
            dtype=float,
        )
        @ np.asarray(
            expected_returns,
            dtype=float,
        )
    )


def calculate_portfolio_volatility(
    weights: pd.Series | np.ndarray,
    covariance_matrix: pd.DataFrame | np.ndarray,
) -> float:
    """
    Calculate model-implied annual portfolio volatility.
    """

    weight_array = np.asarray(
        weights,
        dtype=float,
    )

    covariance_array = np.asarray(
        covariance_matrix,
        dtype=float,
    )

    variance = float(
        weight_array.T
        @ covariance_array
        @ weight_array
    )

    return float(
        np.sqrt(
            max(
                variance,
                0.0,
            )
        )
    )


def calculate_percentage_risk_contributions(
    weights: pd.Series | np.ndarray,
    covariance_matrix: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """
    Calculate each stock's percentage contribution to variance.
    """

    weight_array = np.asarray(
        weights,
        dtype=float,
    )

    covariance_array = np.asarray(
        covariance_matrix,
        dtype=float,
    )

    marginal_variance = (
        covariance_array
        @ weight_array
    )

    total_variance = float(
        weight_array.T
        @ covariance_array
        @ weight_array
    )

    if total_variance <= 0:
        raise ValueError(
            "Portfolio variance must be greater than zero."
        )

    return (
        weight_array
        * marginal_variance
        / total_variance
    )


def calculate_realised_statistics(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    initial_investment_inr: float = 1_000_000,
) -> pd.Series:
    """
    Calculate historical realised portfolio statistics.
    """

    clean_returns = clean_return_data(
        asset_returns
    )

    aligned_weights = (
        pd.Series(
            weights,
            dtype=float,
        )
        .reindex(
            clean_returns.columns
        )
    )

    if aligned_weights.isna().any():
        raise ValueError(
            "Portfolio weights do not match the return columns."
        )

    aligned_weights = (
        aligned_weights
        / aligned_weights.sum()
    )

    portfolio_returns = (
        clean_returns
        @ aligned_weights
    )

    wealth_index = (
        1.0
        + portfolio_returns
    ).cumprod()

    observations = len(
        portfolio_returns
    )

    years = (
        observations
        / TRADING_DAYS_PER_YEAR
    )

    ending_wealth = float(
        wealth_index.iloc[-1]
    )

    cagr = (
        ending_wealth
        ** (
            1.0
            / years
        )
        - 1.0
        if years > 0
        else np.nan
    )

    annualised_return = (
        portfolio_returns.mean()
        * TRADING_DAYS_PER_YEAR
    )

    annualised_volatility = (
        portfolio_returns.std(
            ddof=1
        )
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )

    sharpe_ratio = (
        (
            annualised_return
            - risk_free_rate
        )
        / annualised_volatility
        if annualised_volatility > 0
        else np.nan
    )

    downside_returns = (
        portfolio_returns[
            portfolio_returns < 0
        ]
    )

    downside_deviation = (
        downside_returns.std(
            ddof=1
        )
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
        if len(downside_returns) > 1
        else np.nan
    )

    sortino_ratio = (
        (
            annualised_return
            - risk_free_rate
        )
        / downside_deviation
        if (
            pd.notna(
                downside_deviation
            )
            and downside_deviation > 0
        )
        else np.nan
    )

    running_peak = (
        wealth_index.cummax()
    )

    drawdown = (
        wealth_index
        / running_peak
        - 1.0
    )

    maximum_drawdown = float(
        drawdown.min()
    )

    calmar_ratio = (
        cagr
        / abs(
            maximum_drawdown
        )
        if maximum_drawdown < 0
        else np.nan
    )

    beta = np.nan
    correlation = np.nan

    if benchmark_returns is not None:

        aligned_data = pd.concat(
            [
                portfolio_returns.rename(
                    "Portfolio"
                ),
                benchmark_returns.rename(
                    "Benchmark"
                ),
            ],
            axis=1,
        ).dropna()

        if len(aligned_data) > 1:

            benchmark_variance = (
                aligned_data[
                    "Benchmark"
                ]
                .var(
                    ddof=1
                )
            )

            if benchmark_variance > 0:
                beta = (
                    aligned_data[
                        "Portfolio"
                    ]
                    .cov(
                        aligned_data[
                            "Benchmark"
                        ]
                    )
                    / benchmark_variance
                )

            correlation = (
                aligned_data[
                    "Portfolio"
                ]
                .corr(
                    aligned_data[
                        "Benchmark"
                    ]
                )
            )

    return pd.Series(
        {
            "Observations":
                observations,

            "Ending Value (₹)":
                initial_investment_inr
                * ending_wealth,

            "Total Return":
                ending_wealth
                - 1.0,

            "CAGR":
                cagr,

            "Annualised Volatility":
                annualised_volatility,

            "Sharpe Ratio":
                sharpe_ratio,

            "Sortino Ratio":
                sortino_ratio,

            "Maximum Drawdown":
                maximum_drawdown,

            "Calmar Ratio":
                calmar_ratio,

            "Beta vs Benchmark":
                beta,

            "Correlation vs Benchmark":
                correlation,
        }
    )


def validate_weight_constraints(
    number_of_assets: int,
    minimum_stock_weight: float,
    maximum_stock_weight: float,
) -> None:
    """
    Validate long-only portfolio constraints.
    """

    if minimum_stock_weight < 0:
        raise ValueError(
            "Minimum stock weight cannot be negative."
        )

    if maximum_stock_weight <= 0:
        raise ValueError(
            "Maximum stock weight must be positive."
        )

    if minimum_stock_weight > maximum_stock_weight:
        raise ValueError(
            "Minimum stock weight cannot exceed maximum weight."
        )

    if (
        number_of_assets
        * maximum_stock_weight
        < 1.0 - 1e-10
    ):
        minimum_feasible_cap = (
            1.0
            / number_of_assets
        )

        raise ValueError(
            f"A {maximum_stock_weight:.2%} cap is infeasible "
            f"for {number_of_assets} stocks. The maximum "
            f"weight must be at least "
            f"{minimum_feasible_cap:.2%}."
        )

    if (
        number_of_assets
        * minimum_stock_weight
        > 1.0 + 1e-10
    ):
        raise ValueError(
            "The minimum-weight constraint is infeasible."
        )


def optimise_user_portfolio(
    asset_returns: pd.DataFrame,
    current_weights: pd.Series,
    benchmark_returns: pd.Series | None = None,
    minimum_stock_weight: float = 0.00,
    maximum_stock_weight: float = 0.30,
    expected_return_shrinkage: float = 0.50,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    initial_investment_inr: float = 1_000_000,
) -> dict[str, Any]:
    """
    Construct current, minimum-volatility, maximum-Sharpe
    and risk-parity portfolios.
    """

    clean_returns = clean_return_data(
        asset_returns
    )

    asset_names = (
        clean_returns.columns.tolist()
    )

    number_of_assets = len(
        asset_names
    )

    validate_weight_constraints(
        number_of_assets=number_of_assets,
        minimum_stock_weight=minimum_stock_weight,
        maximum_stock_weight=maximum_stock_weight,
    )

    aligned_current_weights = (
        pd.Series(
            current_weights,
            dtype=float,
        )
        .reindex(
            asset_names
        )
    )

    if aligned_current_weights.isna().any():
        raise ValueError(
            "Current weights do not match the return columns."
        )

    if (
        aligned_current_weights
        < 0
    ).any():
        raise ValueError(
            "Current portfolio weights cannot be negative."
        )

    aligned_current_weights = (
        aligned_current_weights
        / aligned_current_weights.sum()
    )

    (
        expected_returns,
        covariance_matrix,
    ) = estimate_portfolio_inputs(
        asset_returns=clean_returns,
        expected_return_shrinkage=(
            expected_return_shrinkage
        ),
    )

    expected_return_array = (
        expected_returns.to_numpy()
    )

    covariance_array = (
        covariance_matrix.to_numpy()
    )

    starting_weights = np.repeat(
        1.0
        / number_of_assets,
        number_of_assets,
    )

    bounds = [
        (
            minimum_stock_weight,
            maximum_stock_weight,
        )
        for _ in range(
            number_of_assets
        )
    ]

    full_investment_constraint = {
        "type": "eq",
        "fun": lambda weights: (
            np.sum(weights)
            - 1.0
        ),
    }

    def portfolio_return_objective(
        weights,
    ):
        return calculate_portfolio_return(
            weights,
            expected_return_array,
        )

    def portfolio_volatility_objective(
        weights,
    ):
        return calculate_portfolio_volatility(
            weights,
            covariance_array,
        )

    def negative_sharpe_objective(
        weights,
    ):
        volatility = (
            portfolio_volatility_objective(
                weights
            )
        )

        if volatility <= 0:
            return 1e10

        expected_return = (
            portfolio_return_objective(
                weights
            )
        )

        return -(
            (
                expected_return
                - risk_free_rate
            )
            / volatility
        )

    def risk_parity_objective(
        weights,
    ):
        try:
            percentage_contributions = (
                calculate_percentage_risk_contributions(
                    weights,
                    covariance_array,
                )
            )

        except ValueError:
            return 1e10

        target_contribution = (
            1.0
            / number_of_assets
        )

        return float(
            np.sum(
                (
                    percentage_contributions
                    - target_contribution
                )
                ** 2
            )
        )

    def solve_portfolio(
        portfolio_name,
        objective_function,
    ):
        result = minimize(
            fun=objective_function,
            x0=starting_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                full_investment_constraint
            ],
            options={
                "maxiter": 4000,
                "ftol": 1e-12,
                "disp": False,
            },
        )

        if not result.success:
            raise RuntimeError(
                f"{portfolio_name} optimisation failed: "
                f"{result.message}"
            )

        solved_weights = pd.Series(
            result.x,
            index=asset_names,
            name=portfolio_name,
        )

        solved_weights[
            solved_weights.abs()
            < 1e-8
        ] = 0.0

        return (
            solved_weights
            / solved_weights.sum()
        )

    minimum_volatility_weights = (
        solve_portfolio(
            portfolio_name=(
                "Minimum Volatility"
            ),
            objective_function=(
                portfolio_volatility_objective
            ),
        )
    )

    maximum_sharpe_weights = (
        solve_portfolio(
            portfolio_name=(
                "Maximum Sharpe"
            ),
            objective_function=(
                negative_sharpe_objective
            ),
        )
    )

    risk_parity_weights = (
        solve_portfolio(
            portfolio_name=(
                "Risk Parity"
            ),
            objective_function=(
                risk_parity_objective
            ),
        )
    )

    portfolio_weight_sets = {
        "Current Portfolio":
            aligned_current_weights,

        "Minimum Volatility":
            minimum_volatility_weights,

        "Maximum Sharpe":
            maximum_sharpe_weights,

        "Risk Parity":
            risk_parity_weights,
    }

    weight_comparison = pd.DataFrame(
        portfolio_weight_sets
    )

    model_records = []
    realised_records = {}

    for portfolio_name, weights in (
        portfolio_weight_sets.items()
    ):

        expected_return = (
            calculate_portfolio_return(
                weights,
                expected_returns,
            )
        )

        volatility = (
            calculate_portfolio_volatility(
                weights,
                covariance_matrix,
            )
        )

        sharpe_ratio = (
            (
                expected_return
                - risk_free_rate
            )
            / volatility
            if volatility > 0
            else np.nan
        )

        risk_contributions = (
            calculate_percentage_risk_contributions(
                weights,
                covariance_matrix,
            )
        )

        model_records.append(
            {
                "Portfolio":
                    portfolio_name,

                "Expected Return":
                    expected_return,

                "Volatility":
                    volatility,

                "Sharpe Ratio":
                    sharpe_ratio,

                "Largest Weight":
                    weights.max(),

                "Active Holdings":
                    int(
                        (
                            weights
                            > 0.001
                        ).sum()
                    ),

                "Largest Risk Contribution":
                    float(
                        np.max(
                            risk_contributions
                        )
                    ),
            }
        )

        realised_records[
            portfolio_name
        ] = calculate_realised_statistics(
            asset_returns=clean_returns,
            weights=weights,
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
            initial_investment_inr=(
                initial_investment_inr
            ),
        )

    model_comparison = (
        pd.DataFrame(
            model_records
        )
        .set_index("Portfolio")
    )

    realised_comparison = (
        pd.DataFrame(
            realised_records
        )
        .T
    )

    if not np.allclose(
        weight_comparison.sum(
            axis=0
        ),
        1.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Optimised portfolio weights do not sum to 100%."
        )

    optimised_columns = [
        "Minimum Volatility",
        "Maximum Sharpe",
        "Risk Parity",
    ]

    if (
        weight_comparison[
            optimised_columns
        ]
        .max()
        .max()
        > maximum_stock_weight
        + 1e-6
    ):
        raise RuntimeError(
            "An optimised portfolio violates the maximum weight."
        )

    return {
        "weights":
            weight_comparison,

        "model_comparison":
            model_comparison,

        "realised_comparison":
            realised_comparison,

        "expected_returns":
            expected_returns,

        "covariance":
            covariance_matrix,

        "constraints":
            {
                "Minimum Weight":
                    minimum_stock_weight,

                "Maximum Weight":
                    maximum_stock_weight,

                "Holdings":
                    number_of_assets,

                "Expected Return Shrinkage":
                    expected_return_shrinkage,

                "Risk-Free Rate":
                    risk_free_rate,
            },
    }


def build_efficient_frontier(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    comparison_weights: pd.DataFrame | None = None,
    minimum_stock_weight: float = 0.00,
    maximum_stock_weight: float = 0.30,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    frontier_points: int = 60,
) -> dict[str, Any]:
    """
    Build a constrained long-only efficient frontier.
    """

    expected_returns = pd.Series(
        expected_returns,
        dtype=float,
    )

    covariance_matrix = (
        covariance_matrix
        .reindex(
            index=expected_returns.index,
            columns=expected_returns.index,
        )
        .astype(float)
    )

    asset_names = (
        expected_returns.index.tolist()
    )

    number_of_assets = len(
        asset_names
    )

    validate_weight_constraints(
        number_of_assets=number_of_assets,
        minimum_stock_weight=minimum_stock_weight,
        maximum_stock_weight=maximum_stock_weight,
    )

    if frontier_points < 10:
        raise ValueError(
            "At least 10 frontier points are required."
        )

    expected_return_array = (
        expected_returns.to_numpy()
    )

    covariance_array = (
        covariance_matrix.to_numpy()
    )

    starting_weights = np.repeat(
        1.0
        / number_of_assets,
        number_of_assets,
    )

    bounds = [
        (
            minimum_stock_weight,
            maximum_stock_weight,
        )
        for _ in range(
            number_of_assets
        )
    ]

    full_investment_constraint = {
        "type": "eq",
        "fun": lambda weights: (
            np.sum(weights)
            - 1.0
        ),
    }

    def portfolio_return(
        weights,
    ):
        return calculate_portfolio_return(
            weights,
            expected_return_array,
        )

    def portfolio_volatility(
        weights,
    ):
        return calculate_portfolio_volatility(
            weights,
            covariance_array,
        )

    def solve_endpoint(
        objective_function,
        portfolio_name,
    ):
        result = minimize(
            fun=objective_function,
            x0=starting_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                full_investment_constraint
            ],
            options={
                "maxiter": 4000,
                "ftol": 1e-12,
                "disp": False,
            },
        )

        if not result.success:
            raise RuntimeError(
                f"{portfolio_name} failed: "
                f"{result.message}"
            )

        solved_weights = pd.Series(
            result.x,
            index=asset_names,
        )

        solved_weights[
            solved_weights.abs()
            < 1e-8
        ] = 0.0

        return (
            solved_weights
            / solved_weights.sum()
        )

    minimum_volatility_weights = (
        solve_endpoint(
            objective_function=(
                portfolio_volatility
            ),
            portfolio_name=(
                "Minimum-volatility endpoint"
            ),
        )
    )

    maximum_return_weights = (
        solve_endpoint(
            objective_function=lambda weights: (
                -portfolio_return(
                    weights
                )
            ),
            portfolio_name=(
                "Maximum-return endpoint"
            ),
        )
    )

    minimum_feasible_return = (
        portfolio_return(
            minimum_volatility_weights
        )
    )

    maximum_feasible_return = (
        portfolio_return(
            maximum_return_weights
        )
    )

    target_returns = np.linspace(
        minimum_feasible_return,
        maximum_feasible_return
        - 1e-7,
        frontier_points,
    )

    frontier_records = []
    frontier_weight_records = []

    rolling_starting_weights = (
        minimum_volatility_weights.to_numpy()
    )

    for frontier_number, target_return in enumerate(
        target_returns,
        start=1,
    ):

        target_return_constraint = {
            "type": "eq",
            "fun": (
                lambda weights, target=target_return: (
                    portfolio_return(
                        weights
                    )
                    - target
                )
            ),
        }

        result = minimize(
            fun=portfolio_volatility,
            x0=rolling_starting_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                full_investment_constraint,
                target_return_constraint,
            ],
            options={
                "maxiter": 4000,
                "ftol": 1e-12,
                "disp": False,
            },
        )

        if not result.success:
            continue

        frontier_weights = pd.Series(
            result.x,
            index=asset_names,
        )

        frontier_weights[
            frontier_weights.abs()
            < 1e-8
        ] = 0.0

        frontier_weights = (
            frontier_weights
            / frontier_weights.sum()
        )

        expected_return = (
            portfolio_return(
                frontier_weights
            )
        )

        volatility = (
            portfolio_volatility(
                frontier_weights
            )
        )

        sharpe_ratio = (
            (
                expected_return
                - risk_free_rate
            )
            / volatility
            if volatility > 0
            else np.nan
        )

        frontier_records.append(
            {
                "Frontier Portfolio":
                    frontier_number,

                "Expected Return":
                    expected_return,

                "Volatility":
                    volatility,

                "Sharpe Ratio":
                    sharpe_ratio,

                "Largest Weight":
                    frontier_weights.max(),

                "Active Holdings":
                    int(
                        (
                            frontier_weights
                            > 0.001
                        ).sum()
                    ),
            }
        )

        weight_record = {
            "Frontier Portfolio":
                frontier_number,
        }

        weight_record.update(
            frontier_weights.to_dict()
        )

        frontier_weight_records.append(
            weight_record
        )

        rolling_starting_weights = (
            frontier_weights.to_numpy()
        )

    frontier = (
        pd.DataFrame(
            frontier_records
        )
        .set_index(
            "Frontier Portfolio"
        )
    )

    frontier_weights = (
        pd.DataFrame(
            frontier_weight_records
        )
        .set_index(
            "Frontier Portfolio"
        )
    )

    if frontier.empty:
        raise RuntimeError(
            "No efficient-frontier portfolios were generated."
        )

    if len(frontier) < int(
        frontier_points
        * 0.80
    ):
        raise RuntimeError(
            "Too many efficient-frontier optimisations failed."
        )

    best_frontier_index = (
        frontier[
            "Sharpe Ratio"
        ]
        .idxmax()
    )

    comparison_records = []

    if comparison_weights is not None:

        aligned_comparison_weights = (
            comparison_weights
            .reindex(
                index=asset_names
            )
        )

        if aligned_comparison_weights.isna().any().any():
            raise ValueError(
                "Comparison weights do not match the assets."
            )

        for portfolio_name in (
            aligned_comparison_weights.columns
        ):

            weights = (
                aligned_comparison_weights[
                    portfolio_name
                ]
            )

            expected_return = (
                portfolio_return(
                    weights
                )
            )

            volatility = (
                portfolio_volatility(
                    weights
                )
            )

            comparison_records.append(
                {
                    "Portfolio":
                        portfolio_name,

                    "Expected Return":
                        expected_return,

                    "Volatility":
                        volatility,

                    "Sharpe Ratio":
                        (
                            (
                                expected_return
                                - risk_free_rate
                            )
                            / volatility
                            if volatility > 0
                            else np.nan
                        ),
                }
            )

    comparison_points = (
        pd.DataFrame(
            comparison_records
        )
        .set_index("Portfolio")
        if comparison_records
        else pd.DataFrame()
    )

    return {
        "frontier":
            frontier,

        "frontier_weights":
            frontier_weights,

        "best_frontier_result":
            frontier.loc[
                best_frontier_index
            ],

        "best_frontier_weights":
            frontier_weights.loc[
                best_frontier_index
            ],

        "comparison_points":
            comparison_points,

        "minimum_feasible_return":
            minimum_feasible_return,

        "maximum_feasible_return":
            maximum_feasible_return,
    }
