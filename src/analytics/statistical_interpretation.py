"""Structured, evidence-grounded 8-layer statistical interpretation generator."""

from src.analytics.statistical_models import RegressionModelResult, StatisticalTestResult


def render_statistical_interpretation_markdown(result: StatisticalTestResult) -> str:
    """Format the 8-layer interpretation framework into GitHub-style markdown."""
    layers = result.interpretation_layers
    if not layers:
        return f"**Summary:** {result.summary}\n\n**Practical Significance:** {result.practical_significance}"

    sections = []
    layer_headers = [
        ("Research Question", "🎯 1. Research Question"),
        ("Method & Rationale", "📐 2. Method & Rationale"),
        ("Quantitative Result", "📊 3. Quantitative Evidence"),
        ("Statistical Evidence", "⚖️ 4. Statistical Evidence"),
        ("Effect Size & Practical Importance", "📏 5. Effect Size & Practical Importance"),
        ("Assumptions & Limitations", "⚠️ 6. Assumptions & Limitations"),
        ("Conclusion", "💡 7. Grounded Conclusion"),
        ("Next Steps", "🧭 8. Recommended Next Investigation"),
    ]

    for key, title in layer_headers:
        content = layers.get(key, "")
        if content:
            sections.append(f"#### {title}\n{content}\n")

    return "\n".join(sections)


def render_regression_interpretation_markdown(result: RegressionModelResult) -> str:
    """Format regression interpretation framework into structured markdown."""
    layers = result.interpretation_layers
    if not layers:
        return f"**Summary:** {result.summary}"

    sections = []
    layer_headers = [
        ("Research Question", "🎯 1. Model Objective"),
        ("Method & Rationale", "📐 2. Model Specification"),
        ("Quantitative Result", "📊 3. Model Fit & Diagnostics"),
        ("Statistical Evidence", "⚖️ 4. Statistical Evidence"),
        ("Effect Size & Practical Importance", "📏 5. Variance Explained (R²)"),
        ("Assumptions & Limitations", "⚠️ 6. Diagnostic Caveats & Non-Causal Boundary"),
        ("Conclusion", "💡 7. Synthesis"),
        ("Next Steps", "🧭 8. Recommended Refinements"),
    ]

    for key, title in layer_headers:
        content = layers.get(key, "")
        if content:
            sections.append(f"#### {title}\n{content}\n")

    return "\n".join(sections)
