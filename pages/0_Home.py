"""Home: what the platform is, its components, the analytical workflow, and a quick guide."""

import streamlit as st
from src.config.settings import get_settings
from src.core.session_state import KEY_CURRENT_DATASET, init_session_state
from src.core.site_content import GUIDE, MODULES, PRINCIPLES, TRACKS, WORKFLOW
from src.core.ui_components import esc, render_sidebar_status
from src.datascience.catalog import CATALOG
from src.ingestion.loader import ManagedDataset
from src.ingestion.service import SAMPLE_CATALOG, load_sample_dataset

settings = get_settings()

st.set_page_config(
    page_title=f"{settings.app_name} | Home",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_sidebar_status()

TRACK_COLORS = {
    "analyst": "#3b82f6",
    "scientist": "#8b5cf6",
    "shared": "#10b981",
    "workspace": "#06b6d4",
}
STEP_COLORS = ["#3b82f6", "#8b5cf6", "#10b981", "#ef4444", "#f59e0b", "#06b6d4"]

st.markdown(
    """
<style>
.qt-hero { position: relative; overflow: hidden; background: linear-gradient(135deg, rgba(15,23,42,.9), rgba(17,24,52,.85));
    border: 1px solid #25324d; border-radius: 18px; padding: 2.6rem 2.8rem; margin: .4rem 0 .9rem 0; }
.qt-hero::after { content: ""; position: absolute; right: -80px; top: -80px; width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(56,189,248,.28), transparent 65%); }
.qt-hero h1 { margin: 0; font-size: 2.6rem !important; color: #f8fafc; position: relative; }
.qt-tagline { color: #38bdf8; font-size: 1.2rem; font-weight: 600; margin: .7rem 0 .6rem 0; position: relative; }
.qt-hero p { color: #94a3b8; max-width: 56rem; line-height: 1.6; margin: 0; position: relative; }
.qt-tile { background: rgba(15,23,42,.8); border: 1px solid #243049; border-top: 3px solid var(--c);
    border-radius: 12px; padding: 18px 12px; text-align: center; }
.qt-tile b { display: block; font-size: 2.1rem; color: #f1f5f9; line-height: 1.1; }
.qt-tile span { color: #94a3b8; font-size: .72rem; letter-spacing: .09em; text-transform: uppercase; }
.qt-h { font-size: 1.55rem; font-weight: 700; margin: 1.6rem 0 .7rem 0; color: #f1f5f9; }
.qt-card { background: rgba(15,23,42,.78); border: 1px solid #243049; border-top: 3px solid var(--c);
    border-radius: 12px; padding: 14px 16px; min-height: 190px; }
.qt-card h4 { margin: 0 0 .5rem 0; font-size: 1rem; color: #f1f5f9; }
.qt-card p { color: #94a3b8; font-size: .82rem; line-height: 1.5; margin: 0 0 .6rem 0; }
.qt-card ul { margin: 0; padding-left: 1.1rem; color: #cbd5e1; font-size: .82rem; line-height: 1.7; }
.qt-feature { background: rgba(15,23,42,.78); border: 1px solid #243049; border-radius: 12px 12px 0 0;
    padding: 12px 14px 14px 14px; min-height: 270px; }
.qt-feature .art { background: rgba(2,6,23,.7); border-radius: 8px; height: 92px; margin-bottom: 12px;
    display: flex; align-items: center; justify-content: center; }
.qt-feature h4 { margin: 0 0 .4rem 0; font-size: .98rem; color: #f1f5f9; }
.qt-feature p { color: #94a3b8; font-size: .8rem; line-height: 1.5; min-height: 52px; margin: 0 0 .6rem 0; }
.qt-tag { display: inline-block; background: rgba(2,132,199,.18); border: 1px solid #0369a1; color: #7dd3fc;
    border-radius: 4px; font-size: .68rem; padding: 2px 8px; margin: 0 5px 4px 0; }
.qt-panel { background: rgba(15,23,42,.78); border: 1px solid #243049; border-radius: 12px; padding: 18px 22px; }
.qt-row { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #1c2740;
    font-size: .88rem; color: #94a3b8; }
.qt-row b { color: #e2e8f0; font-weight: 600; }
.qt-badge { background: #1d4ed8; color: #fff; border-radius: 6px; padding: 3px 12px; font-size: .78rem; font-weight: 700; }
.qt-step { border-radius: 10px; padding: 14px 12px; text-align: center; min-height: 130px;
    background: color-mix(in srgb, var(--c) 14%, #0b1222); border: 1px solid color-mix(in srgb, var(--c) 45%, #0b1222); }
.qt-step b { color: color-mix(in srgb, var(--c) 55%, #fff); font-size: .88rem; }
.qt-step p { color: #94a3b8; font-size: .75rem; margin: .35rem 0 0 0; line-height: 1.4; }
[data-testid="stPageLink"] a { background: rgba(30,64,175,.28); border: 1px solid #1e40af; border-radius: 8px;
    justify-content: center; padding: .35rem .6rem; }
[data-testid="stPageLink"] a:hover { background: rgba(37,99,235,.4); border-color: #3b82f6; }
</style>
""",
    unsafe_allow_html=True,
)


def heading(text: str) -> None:
    st.markdown(f'<div class="qt-h">{text}</div>', unsafe_allow_html=True)


def art(svg_body: str) -> str:
    return f'<svg viewBox="0 0 200 70" width="200" height="70">{svg_body}</svg>'


ARTS = {
    "quality": art(
        '<rect x="20" y="30" width="18" height="30" fill="#38bdf8"/><rect x="48" y="14" width="18" height="46" fill="#38bdf8"/>'
        '<rect x="76" y="38" width="18" height="22" fill="#f59e0b"/><rect x="104" y="22" width="18" height="38" fill="#38bdf8"/>'
        '<circle cx="165" cy="35" r="22" fill="none" stroke="#10b981" stroke-width="7" stroke-dasharray="105 140"/>'
    ),
    "model": art(
        '<polyline points="10,55 40,48 70,40 100,28 130,22 160,12 190,8" fill="none" stroke="#10b981" stroke-width="3"/>'
        '<polyline points="10,58 70,50 130,40 190,28" fill="none" stroke="#64748b" stroke-width="2" stroke-dasharray="4 4"/>'
        '<circle cx="190" cy="8" r="4" fill="#10b981"/>'
    ),
    "explain": art(
        '<rect x="20" y="8" width="150" height="9" rx="3" fill="#8b5cf6"/><rect x="20" y="24" width="110" height="9" rx="3" fill="#8b5cf6"/>'
        '<rect x="20" y="40" width="70" height="9" rx="3" fill="#a78bfa"/><rect x="20" y="56" width="38" height="9" rx="3" fill="#c4b5fd"/>'
    ),
    "series": art(
        '<polyline points="10,45 30,35 50,42 70,28 90,34 110,24" fill="none" stroke="#e2e8f0" stroke-width="2"/>'
        '<line x1="110" y1="4" x2="110" y2="66" stroke="#ef4444" stroke-dasharray="3 3"/>'
        '<polygon points="110,24 190,6 190,44" fill="#38bdf8" opacity=".35"/>'
    ),
    "chat": art(
        '<rect x="14" y="8" width="110" height="22" rx="8" fill="#1e3a8a"/><rect x="76" y="38" width="110" height="22" rx="8" fill="#0f766e"/>'
        '<circle cx="30" cy="19" r="3" fill="#93c5fd"/><circle cx="42" cy="19" r="3" fill="#93c5fd"/><circle cx="54" cy="19" r="3" fill="#93c5fd"/>'
    ),
}
FEATURED = [
    (
        "Data Quality Lab",
        "Profile, score, and clean data with previewed fixes and a reusable recipe.",
        ["Audit", "Cleaning", "Recipe"],
        "pages/2_Data_Quality.py",
        "quality",
    ),
    (
        "Model Lab",
        "Train, AutoML, race 22 models with nested CV, ensemble, and tune.",
        ["AutoML", "CV", "Ensembles"],
        "pages/6_Model_Lab.py",
        "model",
    ),
    (
        "Model Explainer",
        "Importance, PDP, what-if, counterfactuals, fairness, and drift.",
        ["SHAP", "Fairness", "Drift"],
        "pages/25_Model_Explainer.py",
        "explain",
    ),
    (
        "Time Series",
        "Decompose and forecast with classical or machine-learning models.",
        ["Forecast", "Backtest", "ML"],
        "pages/16_Time_Series_Lab.py",
        "series",
    ),
    (
        "AI Analyst",
        "Ask in plain English; verified tools compute, the AI narrates.",
        ["LLM", "Evidence", "Audit"],
        "pages/5_AI_Analyst.py",
        "chat",
    ),
]

dataset = st.session_state.get(KEY_CURRENT_DATASET)
has_data = isinstance(dataset, ManagedDataset)

# ------------------------------------------------------------------ hero
st.markdown(
    f"""
<div class="qt-hero">
  <h1>📊 {esc(settings.app_name)}</h1>
  <div class="qt-tagline">AI Data Analyst & Data Scientist Platform</div>
  <p>A unified workspace for data quality, exploration, statistical testing, predictive modelling,
  forecasting and reporting. Deterministic engines do the computing, the AI plans and narrates, and
  every result comes with the Python to reproduce it.</p>
</div>
""",
    unsafe_allow_html=True,
)
b1, b2, b3, _ = st.columns([1.2, 1.2, 1.2, 3])
b1.page_link("pages/1_Executive_Overview.py", label="Upload data →", icon="📥")
b2.page_link("pages/7_Intelligence_Reports.py", label="Generate a report", icon="📑")
if b3.button("✨ Try sample data", use_container_width=True):
    load_sample_dataset(st.session_state, settings.max_upload_size_mb)
    st.switch_page("pages/1_Executive_Overview.py")

st.markdown("")
stats = [
    (len(MODULES), "Workspaces", "#3b82f6"),
    (len([m for m in CATALOG if m.available]), "Models in the catalog", "#8b5cf6"),
    (len(SAMPLE_CATALOG) + 1, "Sample datasets", "#10b981"),
    (len(WORKFLOW), "Workflow steps", "#ef4444"),
    (1, "Integrated platform", "#06b6d4"),
]
for col, (value, label, color) in zip(st.columns(len(stats)), stats, strict=True):
    col.markdown(
        f'<div class="qt-tile" style="--c:{color}"><b>{value}</b><span>{label}</span></div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ quick access
heading("⚡ Quick Access")
quick = [
    ("📈", "Overview", "pages/1_Executive_Overview.py"),
    ("🔍", "Data Quality", "pages/2_Data_Quality.py"),
    ("🧭", "Exploration", "pages/3_Exploration_Studio.py"),
    ("🧪", "Statistics", "pages/4_Statistical_Intelligence.py"),
    ("🧠", "Model Lab", "pages/6_Model_Lab.py"),
    ("🤖", "AI Analyst", "pages/5_AI_Analyst.py"),
    ("📑", "Reports", "pages/7_Intelligence_Reports.py"),
]
for col, (icon, label, page) in zip(st.columns(len(quick)), quick, strict=True):
    col.page_link(page, label=label, icon=icon, use_container_width=True)

# ------------------------------------------------------------------ domains
heading("🧭 Analytical Domains")
domain_tracks = [(k, v) for k, v in TRACKS.items() if any(m.track == k for m in MODULES)]
for col, (key, (title, blurb)) in zip(st.columns(len(domain_tracks)), domain_tracks, strict=True):
    items = "".join(f"<li>{m.icon} {esc(m.title)}</li>" for m in MODULES if m.track == key)
    col.markdown(
        f'<div class="qt-card" style="--c:{TRACK_COLORS.get(key, "#3b82f6")}"><h4>{esc(title)}</h4>'
        f"<p>{esc(blurb)}</p><ul>{items}</ul></div>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ featured
heading("⭐ Featured Workspaces")
for col, (title, desc, tags, page, art_key) in zip(
    st.columns(len(FEATURED)), FEATURED, strict=True
):
    with col:
        chips = "".join(f'<span class="qt-tag">{t}</span>' for t in tags)
        st.markdown(
            f'<div class="qt-feature"><div class="art">{ARTS[art_key]}</div><h4>{esc(title)}</h4>'
            f"<p>{esc(desc)}</p>{chips}</div>",
            unsafe_allow_html=True,
        )
        st.page_link(page, label="Open workspace →", use_container_width=True)

# ------------------------------------------------------------------ workflow
st.subheader("Workflow")
st.caption("A typical path from file to report. Jump in at any step.")
PER_ROW = 5
for start in range(0, len(WORKFLOW), PER_ROW):
    for col, step in zip(st.columns(PER_ROW), WORKFLOW[start : start + PER_ROW], strict=False):
        color = STEP_COLORS[(step.number - 1) % len(STEP_COLORS)]
        col.markdown(
            f'<div class="qt-step" style="--c:{color}">{step.icon}<br><b>{step.number}. {esc(step.title)}</b>'
            f"<p>{esc(step.description)}</p></div>",
            unsafe_allow_html=True,
        )
        col.page_link(step.page, label="Go →", use_container_width=True)

# ------------------------------------------------------------------ components (detail)
st.subheader("Components")
st.caption("Everything the platform contains, grouped by role.")
for tab, (track_key, (_title, track_blurb)) in zip(
    st.tabs([t for _, (t, _) in domain_tracks]), domain_tracks, strict=True
):
    with tab:
        st.caption(track_blurb)
        modules = [m for m in MODULES if m.track == track_key]
        for start in range(0, len(modules), 3):
            for col, module in zip(st.columns(3), modules[start : start + 3], strict=False):
                color = TRACK_COLORS.get(track_key, "#3b82f6")
                feats = "".join(f"<li>{esc(f)}</li>" for f in module.features)
                col.markdown(
                    f'<div class="qt-card" style="--c:{color}"><h4>{module.icon} {esc(module.title)}</h4>'
                    f"<p>{esc(module.summary)}</p><ul>{feats}</ul></div>",
                    unsafe_allow_html=True,
                )
                col.page_link(module.page, label="Open →", use_container_width=True)

# ------------------------------------------------------------------ about + guide
st.subheader("About the project")
about_col, principles_col = st.columns([3, 2], gap="large")
with about_col:
    st.markdown(
        "This platform pairs the work of a **data analyst** (cleaning, exploring, and communicating "
        "data) with that of a **data scientist** (testing hypotheses and building validated models), "
        "and puts an **AI agent** on top that can plan analyses and explain results in plain English.\n\n"
        "Deterministic engines do all the computation. The language model only plans and narrates, "
        "so every claim traces back to evidence you can inspect."
    )
with principles_col:
    for title, text in PRINCIPLES:
        st.markdown(
            f"**{title}** · <span class='fcard-body'>{esc(text)}</span>", unsafe_allow_html=True
        )

st.subheader("Quick guide")
guide_col, status_col = st.columns([3, 2], gap="large")
with guide_col:
    for question, answer in GUIDE:
        with st.expander(question):
            st.markdown(answer)
with status_col:
    rows = [
        ("Version", settings.app_version),
        ("Environment", settings.app_env),
        ("Max upload", f"{settings.max_upload_size_mb} MB"),
        ("AI mode", "Gemini" if settings.is_llm_configured else "Offline (no key)"),
    ]
    body = "".join(
        f'<div class="qt-row"><span>{k}</span><b>{esc(str(v))}</b></div>' for k, v in rows
    )
    st.markdown(f'<div class="qt-panel"><b>System status</b>{body}</div>', unsafe_allow_html=True)
