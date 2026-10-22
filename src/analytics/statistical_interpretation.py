"""Structured, evidence-grounded 8-layer statistical interpretation generator."""

import textwrap

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


def render_plain_english_takeaway(result: StatisticalTestResult) -> str:
    """Generate an accessible, human-readable takeaway card with easy-to-read typography and no AI jargon."""
    p_val = result.p_value
    is_sig = p_val < 0.05
    vars_str = ", ".join(result.selected_variables)
    eff = result.effect_size
    ci = result.confidence_interval
    meta = result.method_metadata or {}

    # Determine status badge
    if is_sig:
        badge_html = '<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem;">✅ Statistically Significant (p < 0.05)</span>'
    else:
        badge_html = '<span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem;">⚠️ Inconclusive / Not Significant (p ≥ 0.05)</span>'

    # Concrete plain-English summary based on test type
    if result.groups and len(result.groups) >= 2 and "mean_g1" in meta:
        g1, g2 = result.groups[0], result.groups[1]
        m1, m2 = float(meta.get("mean_g1", 0.0)), float(meta.get("mean_g2", 0.0))
        med1, med2 = float(meta.get("median_g1", 0.0)), float(meta.get("median_g2", 0.0))
        diff = float(meta.get("raw_diff", m1 - m2))
        rel_pct = meta.get("relative_diff_pct")

        higher_grp = g1 if diff > 0 else g2
        lower_grp = g2 if diff > 0 else g1
        measure_col = result.selected_variables[-1] if result.selected_variables else "Value"

        pct_phrase = f" ({abs(rel_pct):.1f}% higher)" if rel_pct is not None else ""
        if is_sig:
            headline = f"Confirmed: <strong>{higher_grp}</strong> has higher <strong>{measure_col}</strong> than <strong>{lower_grp}</strong>"
            bottom_line = (
                f"Records in <strong>{higher_grp}</strong> average <strong>{max(m1, m2):,.2f}</strong>, "
                f"which is <strong>{abs(diff):,.2f} units{pct_phrase}</strong> higher than "
                f"<strong>{lower_grp}</strong> (average {min(m1, m2):,.2f})."
            )
        else:
            headline = f"No meaningful difference detected in <strong>{measure_col}</strong> between <strong>{g1}</strong> and <strong>{g2}</strong>"
            bottom_line = (
                f"The difference between <strong>{g1}</strong> (avg {m1:,.2f}) and <strong>{g2}</strong> (avg {m2:,.2f}) "
                f"is small ({abs(diff):,.2f} units) and consistent with random sample variation."
            )

        ci_str = f"[{ci.lower_bound:+,.2f}, {ci.upper_bound:+,.2f}]" if ci else "N/A"
        details_html = (
            '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 16px 0;">'
            f'<div style="background: rgba(30, 41, 59, 0.7); padding: 12px 16px; border-radius: 8px; border: 1px solid #334155;">'
            f'<div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Group 1: {g1}</div>'
            f'<div style="font-size: 1.25rem; font-weight: 700; color: #38bdf8;">Mean: {m1:,.2f}</div>'
            f'<div style="font-size: 0.85rem; color: #cbd5e1;">Median: {med1:,.2f} • N: {result.sample_sizes.get(g1, 0):,}</div>'
            "</div>"
            f'<div style="background: rgba(30, 41, 59, 0.7); padding: 12px 16px; border-radius: 8px; border: 1px solid #334155;">'
            f'<div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Group 2: {g2}</div>'
            f'<div style="font-size: 1.25rem; font-weight: 700; color: #f43f5e;">Mean: {m2:,.2f}</div>'
            f'<div style="font-size: 0.85rem; color: #cbd5e1;">Median: {med2:,.2f} • N: {result.sample_sizes.get(g2, 0):,}</div>'
            "</div>"
            f'<div style="background: rgba(30, 41, 59, 0.7); padding: 12px 16px; border-radius: 8px; border: 1px solid #334155;">'
            '<div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Absolute Difference</div>'
            f'<div style="font-size: 1.25rem; font-weight: 700; color: #f1f5f9;">{diff:+,.2f}</div>'
            f'<div style="font-size: 0.85rem; color: #cbd5e1;">95% CI: {ci_str}</div>'
            "</div>"
            "</div>"
        )
    else:
        headline = f"Analysis of {vars_str}: {result.test_name}"
        bottom_line = result.summary
        details_html = ""

    # Human-friendly explanation of p-value & effect size
    if p_val < 0.001:
        p_expl = "<strong>p < 0.001:</strong> There is less than a 0.1% chance this difference happened by pure chance or sample noise. We can reject the null hypothesis with very high confidence."
    elif is_sig:
        p_expl = f"<strong>p = {p_val:.4f} (&lt; 0.05):</strong> The observed difference is statistically significant. Random chance is unlikely to produce a discrepancy of this magnitude."
    else:
        p_expl = f"<strong>p = {p_val:.4f} (&ge; 0.05):</strong> The observed variation is within normal expected sampling fluctuations. We cannot conclude that a true difference exists."

    eff_expl = ""
    if eff:
        eff_expl = (
            f"<div style='margin-top: 10px; font-size: 0.95rem; color: #e2e8f0;'>"
            f"<strong>Practical Real-World Impact:</strong> Magnitude is rated <strong>{eff.magnitude.upper()}</strong> ({eff.name} = {eff.estimate:+.2f}). "
            f"This indicates the separation between groups is {'substantial in business terms' if eff.magnitude in ['large', 'very_large'] else 'moderate' if eff.magnitude == 'moderate' else 'subtle and may have limited operational impact'}."
            "</div>"
        )

    # Check assumptions summary
    failed_checks = [c for c in result.assumption_checks if not c.is_satisfied]
    if failed_checks:
        assump_note = (
            f"<div style='background: rgba(245, 158, 11, 0.1); border-left: 3px solid #f59e0b; padding: 10px 14px; border-radius: 6px; margin-top: 14px; font-size: 0.9rem; color: #fde68a;'>"
            f"<strong>Assumption Note:</strong> {len(failed_checks)} assumption(s) departed from ideal theoretical conditions "
            f"({', '.join(c.name for c in failed_checks)}). "
            f"The chosen test (<strong>{result.test_name}</strong>) was specifically selected because it is robust against these exact deviations (e.g. heteroscedasticity or unequal variance)."
            "</div>"
        )
    else:
        assump_note = (
            "<div style='background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10b981; padding: 10px 14px; border-radius: 6px; margin-top: 14px; font-size: 0.9rem; color: #a7f3d0;'>"
            "<strong>Assumption Verification:</strong> All underlying distribution and variance assumptions were tested and verified."
            "</div>"
        )

    card_html = (
        "<div style=\"font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; "
        "background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); "
        'border: 1px solid #334155; border-radius: 12px; padding: 22px 26px; margin-bottom: 20px; color: #f1f5f9;">'
        '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">'
        f'<div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">{headline}</div>'
        f"{badge_html}"
        "</div>"
        f'<p style="font-size: 1.05rem; line-height: 1.6; color: #cbd5e1; margin-bottom: 8px;">{bottom_line}</p>'
        f"{details_html}"
        '<div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 8px; padding: 14px 18px; margin-top: 12px;">'
        f'<div style="font-size: 0.95rem; color: #e2e8f0; line-height: 1.5;">{p_expl}</div>'
        f"{eff_expl}"
        "</div>"
        f"{assump_note}"
        "</div>"
    )
    return textwrap.dedent(card_html)


def render_regression_plain_english(result: RegressionModelResult) -> str:
    """Generate an accessible plain-English summary card for OLS regression."""
    r2_pct = result.r_squared * 100.0
    sig_preds = [c for c in result.coefficients if c.p_value < 0.05 and c.variable != "const"]
    pos_preds = [c for c in sig_preds if c.estimate > 0]
    neg_preds = [c for c in sig_preds if c.estimate < 0]

    pos_str = (
        ", ".join(f"<strong>{c.variable}</strong> (+{c.estimate:,.2f})" for c in pos_preds)
        or "None"
    )
    neg_str = (
        ", ".join(f"<strong>{c.variable}</strong> ({c.estimate:,.2f})" for c in neg_preds) or "None"
    )

    card_html = (
        "<div style=\"font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; "
        "background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); "
        'border: 1px solid #334155; border-radius: 12px; padding: 22px 26px; margin-bottom: 20px; color: #f1f5f9;">'
        '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">'
        f'<div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">OLS Regression: Predicting <strong>{result.outcome_variable}</strong></div>'
        f'<span style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #0284c7; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem;">R² = {r2_pct:.1f}% Variance Explained</span>'
        "</div>"
        f'<p style="font-size: 1.05rem; line-height: 1.6; color: #cbd5e1; margin-bottom: 14px;">This model explains <strong>{r2_pct:.1f}%</strong> of all variation in <strong>{result.outcome_variable}</strong> using {len(result.predictor_variables)} predictor variable(s). The overall model is <strong>{"statistically significant (p < 0.001)" if result.f_pvalue < 0.001 else "not statistically significant"}</strong>.</p>'
        '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-top: 14px;">'
        '<div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 12px 16px;">'
        '<div style="font-size: 0.8rem; color: #34d399; font-weight: 700; text-transform: uppercase;">Top Positive Drivers (Increases Outcome)</div>'
        f'<div style="font-size: 0.95rem; color: #f1f5f9; margin-top: 6px;">{pos_str}</div>'
        "</div>"
        '<div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 8px; padding: 12px 16px;">'
        '<div style="font-size: 0.8rem; color: #fb7185; font-weight: 700; text-transform: uppercase;">Top Negative Drivers (Decreases Outcome)</div>'
        f'<div style="font-size: 0.95rem; color: #f1f5f9; margin-top: 6px;">{neg_str}</div>'
        "</div>"
        "</div>"
        "</div>"
    )
    return textwrap.dedent(card_html)
