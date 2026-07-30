import streamlit as st


st.set_page_config(
    page_title="Bharat Portfolio Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


pages = {
    "Platform": [
        st.Page(
            "app_pages/home.py",
            title="Home",
            icon="🏠",
            default=True,
        ),
    ],

    "Portfolio Research": [
        st.Page(
            "app_pages/strategy_dashboard.py",
            title="Strategy Dashboard",
            icon="📊",
        ),

        st.Page(
            "app_pages/robustness_attribution.py",
            title="Robustness & Attribution",
            icon="🧪",
        ),

        st.Page(
            "app_pages/forecasting_risk.py",
            title="Forecasting & Risk Models",
            icon="📈",
        ),


        st.Page(
            "app_pages/optimisation_stress.py",
            title="Optimisation & Stress Testing",
            icon="⚖️",
        ),


        st.Page(
            "app_pages/user_portfolio_lab.py",
            title="User Portfolio Lab",
            icon="🧮",
        ),
    ],
}


selected_page = st.navigation(
    pages,
    position="sidebar",
)

selected_page.run()
