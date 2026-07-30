import streamlit as st


st.title("📊 Bharat Portfolio Lab")

st.subheader(
    "Indian Equity Portfolio Analytics "
    "and Strategy Research Platform"
)

st.markdown(
    """
    Bharat Portfolio Lab is an academic financial-markets analytics
    project designed for Indian listed equities.

    The current research case is the **India 10 Adaptive Barbell
    Portfolio**, consisting of five Resilient Compounders and five
    Growth Leaders.
    """
)

st.info(
    "Use the navigation menu to open each research module."
)

module_col_1, module_col_2, module_col_3 = (
    st.columns(3)
)

with module_col_1:

    st.markdown("### 📊 Strategy Dashboard")

    st.markdown(
        """
        Review portfolio construction, historical performance,
        market-regime exposure and benchmark comparison.
        """
    )

with module_col_2:

    st.markdown("### 🧪 Robustness & Attribution")

    st.markdown(
        """
        Review parameter sensitivity, transaction costs, market
        states, return attribution and overfitting evidence.
        """
    )

with module_col_3:

    st.markdown("### 📈 Forecasting & Risk")

    st.markdown(
        """
        Review ARIMA forecasts, GJR-GARCH volatility estimates,
        model diagnostics and allocation-model comparisons.
        """
    )

st.markdown("---")

st.markdown(
    """
    ### Current research conclusion

    **Moderately robust — strong risk-control evidence, but meaningful
    stock-selection bias and return-concentration limitations remain.**

    The v0.4 forecasting research also found that dynamic volatility
    models are useful for risk monitoring, but did not improve portfolio
    allocation relative to the simpler 63-day historical-volatility
    method.
    """
)

st.warning(
    "This application is an academic financial-analytics project "
    "and does not constitute investment advice."
)
