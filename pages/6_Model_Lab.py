"""Page 6: Model Lab - train, auto-build, compare, ensemble, and tune models."""

from src.sections import render_sections

render_sections(
    "Model Lab",
    "🧠",
    "Everything for building a predictive model, from a quick train to AutoML, model races, ensembles, and tuning.",
    {
        "🔮 Train & evaluate": "predictive_lab",
        "🤖 AutoML": "automl",
        "🧭 Selection & race": "model_selection",
        "🎼 Ensembles": "ensembles",
        "🎛️ Tuning": "model_tuning",
    },
    "model_lab",
)
