# Bharat Portfolio Lab

An explainable Indian equity portfolio analytics and strategy-backtesting platform built using Python, Google Colab and Streamlit.

## Live Application

[Open Bharat Portfolio Lab](https://bharat-portfolio-lab.streamlit.app/)

## Current Version: v0.1

The preliminary version includes:

- India 10 Adaptive Barbell flagship portfolio
- Five Resilient Compounders and five Growth Leaders
- Nifty 50 benchmark comparison
- CAGR and total-return analysis
- Annualised volatility
- Sharpe ratio
- Maximum drawdown analysis
- Portfolio holdings and sector classification
- Interactive investment amount and analysis-period inputs

## Strategy Concept

The flagship portfolio combines:

- **Resilient Compounders:** established companies selected for stability and business quality
- **Growth Leaders:** companies exposed to structural Indian growth themes

The analytical notebook also contains a preliminary rules-based Adaptive Barbell strategy using momentum, inverse-volatility allocation and a Nifty 50 trend-based defensive overlay.

## Technology

- Python
- Pandas
- NumPy
- Matplotlib
- yFinance
- Google Colab
- Streamlit
- GitHub

## Development Roadmap

Future versions will add:

- Adaptive strategy backtesting within the deployed application
- Transaction-cost and turnover analysis
- Portfolio optimisation
- ARIMA and GARCH forecasting
- Monte Carlo stress testing
- Machine-learning model comparison
- Indian yield-curve analytics
- Pairs trading
- User-uploaded portfolio analysis

## Disclaimer

This project is intended for academic and educational purposes and does not constitute investment advice.
<!-- V0.5-RELEASE-CANDIDATE -->

## v0.5 Release Candidate — User Portfolio Lab

Bharat Portfolio Lab v0.5 introduces an interactive portfolio-analysis workflow for Indian listed equities.

Users can enter NSE or BSE holdings using weights, invested amounts or share quantities and run:

- Portfolio performance and diversification analysis
- Minimum-volatility, maximum-Sharpe and risk-parity optimisation
- Efficient-frontier analysis
- Historical stress testing
- VaR and Expected Shortfall
- Block-bootstrap Monte Carlo simulation
- Bull, base and bear scenarios
- Downloadable analysis datasets

BSE quantity valuation uses available BSE prices. When sufficient BSE historical data is unavailable, the corresponding NSE history may be used for return analytics with a visible disclosure.

Read the complete [v0.5.0 release candidate notes](docs/releases/v0.5.0.md).

> v0.5 remains on its feature branch until final review, merge and deployment.
