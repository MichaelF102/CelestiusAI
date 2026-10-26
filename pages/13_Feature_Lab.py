"""Page 13: Feature Lab - create model-ready features, then choose which to keep."""

from src.sections import render_sections

render_sections(
    "Feature Lab",
    "🛠️",
    "Create features with leakage checks, then select the ones that matter.",
    {"🛠️ Engineering": "feature_engineering", "🧹 Selection": "feature_selection"},
    "features",
)
