"""Map the former standalone page files to their merged page and section."""

from streamlit.testing.v1 import AppTest

SECTIONS = {
    "14_Experiments.py": ("4_Statistical_Intelligence.py", "stats", "🧬 A/B testing & power"),
    "13_Feature_Engineering.py": ("13_Feature_Lab.py", "features", "🛠️ Engineering"),
    "24_Feature_Selection.py": ("13_Feature_Lab.py", "features", "🧹 Selection"),
    "16_Time_Series_Lab.py": ("16_Time_Series_Lab.py", "ts", "⏱️ Classical"),
    "28_ML_Forecasting.py": ("16_Time_Series_Lab.py", "ts", "🔮 ML forecasting"),
    "29_Causal_Uplift.py": ("29_Specialised_Methods.py", "special", "🧭 Causal & uplift"),
    "30_Survival_Lab.py": ("29_Specialised_Methods.py", "special", "⏳ Survival"),
    "31_Text_Lab.py": ("29_Specialised_Methods.py", "special", "💬 Text"),
    "6_Predictive_Lab.py": ("6_Model_Lab.py", "model_lab", "🔮 Train & evaluate"),
    "27_AutoML_Lab.py": ("6_Model_Lab.py", "model_lab", "🤖 AutoML"),
    "20_Model_Selection.py": ("6_Model_Lab.py", "model_lab", "🧭 Selection & race"),
    "21_Ensembles.py": ("6_Model_Lab.py", "model_lab", "🎼 Ensembles"),
    "17_Model_Tuning.py": ("6_Model_Lab.py", "model_lab", "🎛️ Tuning"),
    "22_Classification_Toolkit.py": (
        "22_Classification_Regression.py",
        "clf_reg",
        "🎯 Classification",
    ),
    "23_Regression_Workbench.py": ("22_Classification_Regression.py", "clf_reg", "📉 Regression"),
    "26_Experiment_Tracker.py": ("26_MLOps.py", "mlops", "📒 Experiments & registry"),
    "18_Model_Deployment.py": ("26_MLOps.py", "mlops", "🚀 Deployment & monitoring"),
}


def open_page(at: AppTest, page: str) -> AppTest:
    """Open a page by its former file name, selecting the right section when merged."""
    new, key, label = SECTIONS.get(page, (page, None, None))
    at.switch_page(f"pages/{new}").run()
    if key and not at.exception:
        at.radio(key=f"section_{key}").set_value(label).run()
    return at
