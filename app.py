"""CelestiusAI - application entry point and navigation router."""

import streamlit as st
from src.config.settings import get_settings
from src.core.logging_config import setup_logging
from src.core.session_state import init_session_state

settings = get_settings()
setup_logging(settings.log_level)
init_session_state()

navigation = st.navigation(
    {
        "Start": [
            st.Page("pages/0_Home.py", title="Home", icon="🏠", default=True),
            st.Page("pages/1_Executive_Overview.py", title="Executive Overview", icon="📈"),
        ],
        "Data Analyst": [
            st.Page("pages/2_Data_Quality.py", title="Data Quality", icon="🔍"),
            st.Page("pages/9_Data_Dictionary.py", title="Data Dictionary", icon="📖"),
            st.Page("pages/10_Data_Sources.py", title="Data Sources", icon="🔌"),
            st.Page("pages/11_Transformation_Studio.py", title="Transformation Studio", icon="🧱"),
            st.Page("pages/3_Exploration_Studio.py", title="Exploration Studio", icon="🧭"),
            st.Page("pages/12_Dashboard_Builder.py", title="Dashboard Builder", icon="🧩"),
        ],
        "Data Scientist": [
            st.Page(
                "pages/4_Statistical_Intelligence.py", title="Statistical Intelligence", icon="🧪"
            ),
            st.Page("pages/13_Feature_Lab.py", title="Feature Lab", icon="🛠️"),
            st.Page("pages/15_Unsupervised_Lab.py", title="Unsupervised Lab", icon="🔭"),
            st.Page("pages/16_Time_Series_Lab.py", title="Time Series", icon="⏱️"),
            st.Page("pages/29_Specialised_Methods.py", title="Specialised Methods", icon="🧭"),
        ],
        "Modeling": [
            st.Page("pages/6_Model_Lab.py", title="Model Lab", icon="🧠"),
            st.Page(
                "pages/22_Classification_Regression.py",
                title="Classification & Regression",
                icon="🎯",
            ),
            st.Page("pages/25_Model_Explainer.py", title="Model Explainer", icon="💡"),
            st.Page("pages/26_MLOps.py", title="MLOps", icon="📒"),
        ],
        "AI & Reports": [
            st.Page("pages/5_AI_Analyst.py", title="AI Analyst", icon="🤖"),
            st.Page("pages/7_Intelligence_Reports.py", title="Intelligence Reports", icon="📑"),
        ],
        "Workspace": [
            st.Page("pages/8_Projects_Sessions.py", title="Projects & Sessions", icon="💾"),
            st.Page("pages/19_Settings.py", title="Settings", icon="⚙️"),
        ],
    }
)
navigation.run()
