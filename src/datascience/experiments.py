"""A/B test planning and analysis: power, frequentist and Bayesian results, SRM, peeking."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

from src.core.code_snippets import make_snippet


class ExperimentError(ValueError):
    """Invalid experiment input (message is user-facing)."""


# --------------------------------------------------------------------------- planning
def sample_size_proportions(
    baseline: float, target: float, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0
) -> int:
    """Rows needed in the control group to detect baseline -> target (two-sided)."""
    if not (0 < baseline < 1 and 0 < target < 1) or baseline == target:
        raise ExperimentError("Rates must be between 0 and 1 and different from each other.")
    effect = abs(proportion_effectsize(target, baseline))
    n = NormalIndPower().solve_power(effect_size=effect, alpha=alpha, power=power, ratio=ratio)
    return int(np.ceil(n))


def sample_size_means(
    delta: float, sd: float, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0
) -> int:
    if sd <= 0 or delta == 0:
        raise ExperimentError("Standard deviation must be positive and the difference non-zero.")
    n = TTestIndPower().solve_power(
        effect_size=abs(delta) / sd, alpha=alpha, power=power, ratio=ratio
    )
    return int(np.ceil(n))


def power_for_proportions(
    baseline: float, target: float, n_per_group: int, alpha: float = 0.05
) -> float:
    effect = abs(proportion_effectsize(target, baseline))
    return float(
        NormalIndPower().power(effect_size=effect, nobs1=n_per_group, alpha=alpha, ratio=1.0)
    )


# --------------------------------------------------------------------------- analysis
def analyze_conversion(
    control_successes: int,
    control_n: int,
    treatment_successes: int,
    treatment_n: int,
    alpha: float = 0.05,
    simulations: int = 100_000,
    seed: int = 42,
) -> dict[str, Any]:
    for s, n in ((control_successes, control_n), (treatment_successes, treatment_n)):
        if n <= 0 or not 0 <= s <= n:
            raise ExperimentError("Successes must be between 0 and the group size.")
    p_c, p_t = control_successes / control_n, treatment_successes / treatment_n
    diff = p_t - p_c
    se = np.sqrt(p_c * (1 - p_c) / control_n + p_t * (1 - p_t) / treatment_n)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    stat, p_value = proportions_ztest(
        [treatment_successes, control_successes], [treatment_n, control_n]
    )

    rng = np.random.default_rng(seed)
    post_c = rng.beta(1 + control_successes, 1 + control_n - control_successes, simulations)
    post_t = rng.beta(1 + treatment_successes, 1 + treatment_n - treatment_successes, simulations)
    rel = (post_t - post_c) / np.maximum(post_c, 1e-12)
    return {
        "type": "conversion",
        "control_rate": p_c,
        "treatment_rate": p_t,
        "abs_lift": diff,
        "rel_lift": diff / p_c if p_c > 0 else float("nan"),
        "ci_low": diff - z_crit * se,
        "ci_high": diff + z_crit * se,
        "z": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "prob_treatment_better": float(np.mean(post_t > post_c)),
        "expected_loss_if_ship": float(np.mean(np.maximum(post_c - post_t, 0))),
        "rel_lift_credible": tuple(np.percentile(rel, [2.5, 97.5]).tolist()),
        "alpha": alpha,
    }


def analyze_continuous(
    control: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    alpha: float = 0.05,
    bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    a = np.asarray(pd.Series(control).dropna(), dtype=float)
    b = np.asarray(pd.Series(treatment).dropna(), dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ExperimentError("Each group needs at least 2 observations.")
    t_res = stats.ttest_ind(b, a, equal_var=False)
    diff = float(b.mean() - a.mean())
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df_w = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, df_w)
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    )
    d = diff / pooled if pooled > 0 else float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(b, (bootstrap, len(b))).mean(axis=1) - rng.choice(
        a, (bootstrap, len(a))
    ).mean(axis=1)
    return {
        "type": "continuous",
        "control_mean": float(a.mean()),
        "treatment_mean": float(b.mean()),
        "abs_lift": diff,
        "rel_lift": diff / a.mean() if a.mean() != 0 else float("nan"),
        "ci_low": diff - t_crit * se,
        "ci_high": diff + t_crit * se,
        "t": float(t_res.statistic),
        "p_value": float(t_res.pvalue),
        "mann_whitney_p": float(stats.mannwhitneyu(b, a).pvalue),
        "cohens_d": float(d),
        "significant": bool(t_res.pvalue < alpha),
        "prob_treatment_better": float(np.mean(boots > 0)),
        "alpha": alpha,
        "n_control": len(a),
        "n_treatment": len(b),
    }


def analyze_frame(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    control: Any,
    treatment: Any,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run the right analysis for a binary (0/1) or continuous metric."""
    a = df.loc[df[group_col] == control, metric_col].dropna()
    b = df.loc[df[group_col] == treatment, metric_col].dropna()
    if a.empty or b.empty:
        raise ExperimentError("One of the groups has no rows with a metric value.")
    values = set(pd.concat([a, b]).unique())
    if values <= {0, 1}:
        return analyze_conversion(int(a.sum()), len(a), int(b.sum()), len(b), alpha)
    return analyze_continuous(a, b, alpha)


def srm_check(
    counts: list[int], expected_ratios: list[float] | None = None, threshold: float = 0.001
) -> dict[str, Any]:
    """Sample-ratio mismatch: are group sizes consistent with the intended split?"""
    counts_arr = np.asarray(counts, dtype=float)
    if len(counts_arr) < 2 or counts_arr.sum() <= 0:
        raise ExperimentError("Need at least two groups with rows.")
    ratios = np.asarray(expected_ratios or [1] * len(counts_arr), dtype=float)
    expected = ratios / ratios.sum() * counts_arr.sum()
    chi2, p = stats.chisquare(counts_arr, expected)
    return {"chi2": float(chi2), "p_value": float(p), "mismatch": bool(p < threshold)}


# --------------------------------------------------------------------------- peeking
POCOCK_NOMINAL_ALPHA_05 = {
    2: 0.0294,
    3: 0.0221,
    4: 0.0182,
    5: 0.0158,
    6: 0.0142,
    7: 0.0130,
    8: 0.0120,
    9: 0.0112,
    10: 0.0106,
}


def adjusted_alpha(alpha: float, looks: int, method: str = "bonferroni") -> float:
    if looks < 1:
        raise ExperimentError("Planned looks must be at least 1.")
    if looks == 1:
        return alpha
    if method == "pocock" and abs(alpha - 0.05) < 1e-9 and looks in POCOCK_NOMINAL_ALPHA_05:
        return POCOCK_NOMINAL_ALPHA_05[looks]
    return alpha / looks


def simulate_peeking(
    looks: int, n_per_look: int = 200, alpha: float = 0.05, sims: int = 2000, seed: int = 0
) -> dict[str, float]:
    """False-positive rate when an A/A test is checked after every batch and stopped at p < alpha."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(sims, looks * n_per_look))
    b = rng.normal(size=(sims, looks * n_per_look))
    hits_any = np.zeros(sims, dtype=bool)
    for k in range(1, looks + 1):
        n = k * n_per_look
        diff = b[:, :n].mean(axis=1) - a[:, :n].mean(axis=1)
        z = diff / np.sqrt(2.0 / n)
        hits_any |= np.abs(z) > stats.norm.ppf(1 - alpha / 2)
    final_n = looks * n_per_look
    z_final = (b.mean(axis=1) - a.mean(axis=1)) / np.sqrt(2.0 / final_n)
    return {
        "peeking_false_positive_rate": float(hits_any.mean()),
        "single_look_false_positive_rate": float(
            (np.abs(z_final) > stats.norm.ppf(1 - alpha / 2)).mean()
        ),
    }


# --------------------------------------------------------------------------- code
def ab_test_snippet(
    group_col: str, metric_col: str, control: Any, treatment: Any, binary: bool, **load: Any
) -> str:
    if binary:
        analysis = """a = data.loc[data[group_col] == control, metric_col]
b = data.loc[data[group_col] == treatment, metric_col]
from statsmodels.stats.proportion import proportions_ztest
z, p = proportions_ztest([b.sum(), a.sum()], [len(b), len(a)])
p_c, p_t = a.mean(), b.mean()
se = np.sqrt(p_c * (1 - p_c) / len(a) + p_t * (1 - p_t) / len(b))
print(f"control={p_c:.4f} treatment={p_t:.4f} lift={p_t - p_c:+.4f} ({(p_t - p_c) / p_c:+.1%})")
print(f"95% CI of difference: ({p_t - p_c - 1.96 * se:.4f}, {p_t - p_c + 1.96 * se:.4f}), z={z:.3f}, p={p:.4g}")

# %%
# Bayesian view: probability that treatment beats control (uniform Beta(1,1) priors)
rng = np.random.default_rng(42)
post_c = rng.beta(1 + a.sum(), 1 + len(a) - a.sum(), 100_000)
post_t = rng.beta(1 + b.sum(), 1 + len(b) - b.sum(), 100_000)
print("P(treatment > control) =", round((post_t > post_c).mean(), 4))"""
    else:
        analysis = """a = data.loc[data[group_col] == control, metric_col].astype(float)
b = data.loc[data[group_col] == treatment, metric_col].astype(float)
from scipy import stats
t, p = stats.ttest_ind(b, a, equal_var=False)      # Welch
pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
print(f"control mean={a.mean():.4f} treatment mean={b.mean():.4f} lift={b.mean() - a.mean():+.4f}")
print(f"Welch t={t:.3f}, p={p:.4g}, Cohen's d={(b.mean() - a.mean()) / pooled:.3f}")
print("Mann-Whitney p:", round(stats.mannwhitneyu(b, a).pvalue, 4))"""
    body = f"""group_col, metric_col = {group_col!r}, {metric_col!r}
control, treatment = {control!r}, {treatment!r}
data = df[[group_col, metric_col]].dropna()

# Sample-ratio check: are the group sizes what you intended?
from scipy import stats as _st
counts = data[group_col].value_counts()
print(counts.to_dict(), "SRM chi-square p =", round(_st.chisquare(counts.values).pvalue, 4))

# %%
{analysis}
"""
    return make_snippet("A/B test analysis", body, **load)


def power_snippet(baseline: float, target: float, alpha: float = 0.05, power: float = 0.8) -> str:
    body = f"""from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

baseline, target, alpha, power = {baseline}, {target}, {alpha}, {power}
effect = abs(proportion_effectsize(target, baseline))
n = NormalIndPower().solve_power(effect_size=effect, alpha=alpha, power=power, ratio=1.0)
print(f"Rows needed per group: {{int(np.ceil(n)):,}}")
"""
    return make_snippet("Sample size for a conversion test", body, include_load=False)
