"""Portfolio construction, optimisation and risk modules."""

from src.portfolio.optimisation import (
    build_efficient_frontier,
    calculate_percentage_risk_contributions,
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    calculate_realised_statistics,
    estimate_portfolio_inputs,
    optimise_user_portfolio,
)

from src.portfolio.risk import (
    DEFAULT_SCENARIO_ASSUMPTIONS,
    analyse_user_portfolio_risk,
    calculate_diversification_statistics,
    calculate_var_and_expected_shortfall,
    identify_historical_stress_periods,
    run_historical_stress_test,
    run_monte_carlo_risk_analysis,
    run_scenario_analysis,
)

from src.portfolio.user_portfolio import (
    construct_user_portfolio,
    download_latest_stock_prices,
    download_user_market_data,
    standardise_indian_ticker,
    validate_analysis_dates,
)


__all__ = [
    "DEFAULT_SCENARIO_ASSUMPTIONS",
    "analyse_user_portfolio_risk",
    "build_efficient_frontier",
    "calculate_diversification_statistics",
    "calculate_percentage_risk_contributions",
    "calculate_portfolio_return",
    "calculate_portfolio_volatility",
    "calculate_realised_statistics",
    "calculate_var_and_expected_shortfall",
    "construct_user_portfolio",
    "download_latest_stock_prices",
    "download_user_market_data",
    "estimate_portfolio_inputs",
    "identify_historical_stress_periods",
    "optimise_user_portfolio",
    "run_historical_stress_test",
    "run_monte_carlo_risk_analysis",
    "run_scenario_analysis",
    "standardise_indian_ticker",
    "validate_analysis_dates",
]
