"""Static content for the Home page: components, workflow, and user guide."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleInfo:
    track: str  # "analyst" | "scientist" | "shared"
    icon: str
    title: str
    page: str
    summary: str
    features: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowStep:
    number: int
    icon: str
    title: str
    description: str
    page: str


TRACKS = {
    "analyst": (
        "📊 AI Data Analyst",
        "Understand the data: quality, structure, patterns, and what to tell stakeholders.",
    ),
    "scientist": (
        "🧪 AI Data Scientist",
        "Test hypotheses, run experiments, and build validated models.",
    ),
    "shared": ("🤖 AI & Reports", "Ask questions in plain English and share verified results."),
    "workspace": ("💾 Workspace", "Persist your work and control the AI and privacy settings."),
}


def _m(track, icon, title, page, summary, features):  # noqa: ANN001, ANN202
    return ModuleInfo(track, icon, title, page, summary, tuple(features))


MODULES: tuple[ModuleInfo, ...] = (
    _m(
        "analyst",
        "📈",
        "Executive Overview",
        "pages/1_Executive_Overview.py",
        "Upload a file and get a one-screen briefing.",
        [
            "Search-bar upload for CSV and Excel",
            "Severity-ranked key findings",
            "Context-aware next steps",
            "One-click Markdown summary",
        ],
    ),
    _m(
        "analyst",
        "🔍",
        "Data Quality Lab",
        "pages/2_Data_Quality.py",
        "Profile, audit, and clean the data with a full audit trail.",
        [
            "Schema inference and column drill-down",
            "Missing values, duplicates, and outliers",
            "Dry-run previews, bulk apply, and rollback",
            "Replayable cleaning recipes",
        ],
    ),
    _m(
        "analyst",
        "📖",
        "Data Dictionary",
        "pages/9_Data_Dictionary.py",
        "Document what every column means.",
        [
            "Editable descriptions, roles, and units",
            "Business glossary",
            "Shared with the AI as wording only, never values",
            "Export to Markdown or CSV",
        ],
    ),
    _m(
        "analyst",
        "🔌",
        "Data Sources",
        "pages/10_Data_Sources.py",
        "Bring data in from more places.",
        [
            "Read-only SQL over your tables or a SQLite file",
            "Plain-English to SQL, shown before it runs",
            "URLs and Google Sheets with SSRF protection",
            "Side tables for joins",
        ],
    ),
    _m(
        "analyst",
        "🧱",
        "Transformation Studio",
        "pages/11_Transformation_Studio.py",
        "Combine and reshape multiple tables.",
        [
            "Joins, stacking, pivots, and unpivots",
            "Group-by and derived columns",
            "Preview, timeline, and rollback",
            "Replayable pandas code",
        ],
    ),
    _m(
        "analyst",
        "🧭",
        "Exploration Studio",
        "pages/3_Exploration_Studio.py",
        "Explore distributions and relationships with grounded findings.",
        [
            "Distributions, correlations, and category associations",
            "Grouped disparity and time trends",
            "Custom chart builder with pinned visuals",
            "Automatic chart recommendations",
        ],
    ),
    _m(
        "analyst",
        "🧩",
        "Dashboard Builder",
        "pages/12_Dashboard_Builder.py",
        "Compose KPIs and charts with shared filters.",
        [
            "17 tile types: KPIs with trends, gauges, combo, funnel, map and more",
            "Sections, resizable tiles, themes, and colour-blind-safe palettes",
            "Cross-filtering, drill-down, parameters, and calculated fields",
            "Export to HTML, Excel, or PDF; rebuild as a Streamlit app",
        ],
    ),
    _m(
        "scientist",
        "🧪",
        "Statistical Intelligence",
        "pages/4_Statistical_Intelligence.py",
        "Rigorous inference and A/B experiments, explained in plain English.",
        [
            "Hypothesis tests with assumption checks and effect sizes",
            "Multiple-testing correction and OLS diagnostics",
            "A/B sample size, Bayesian results, sample-ratio check, peeking guard",
        ],
    ),
    _m(
        "scientist",
        "🛠️",
        "Feature Lab",
        "pages/13_Feature_Lab.py",
        "Create model-ready columns, then choose which to keep.",
        [
            "Transforms, encodings, bins, dates, interactions with leakage warnings",
            "Filter, wrapper, embedded, and stability selection; VIF and PCA",
            "Reusable JSON recipe and pandas code for every step",
        ],
    ),
    _m(
        "scientist",
        "🔭",
        "Unsupervised Lab",
        "pages/15_Unsupervised_Lab.py",
        "Find structure without a target.",
        [
            "K-means with elbow and silhouette; plain-English cluster profiles",
            "PCA with loadings",
            "Isolation-forest anomalies",
        ],
    ),
    _m(
        "scientist",
        "⏱️",
        "Time Series",
        "pages/16_Time_Series_Lab.py",
        "Understand and forecast time-based data.",
        [
            "Decomposition, stationarity tests, ACF/PACF",
            "Rolling-origin backtests against naive baselines",
            "Machine-learning forecasting on lag features with forecast ranges",
        ],
    ),
    _m(
        "scientist",
        "🧭",
        "Specialised Methods",
        "pages/29_Specialised_Methods.py",
        "Methods for particular questions.",
        [
            "Causal: difference-in-differences, propensity matching, uplift",
            "Survival: Kaplan-Meier, log-rank, Cox model",
            "Text: TF-IDF classification with top terms per class",
        ],
    ),
    _m(
        "scientist",
        "🧠",
        "Model Lab",
        "pages/6_Model_Lab.py",
        "Build predictive models without data leakage.",
        [
            "Train and evaluate with baselines and walk-forward validation",
            "AutoML pipeline with every decision explained",
            "22-model catalog: race, nested CV, statistical comparison, ensembles, tuning (random, halving, Bayesian)",
        ],
    ),
    _m(
        "scientist",
        "🎯",
        "Classification & Regression",
        "pages/22_Classification_Regression.py",
        "Deep-dive toolkits for each task type.",
        [
            "Classification: imbalance, thresholds, calibration, multi-class",
            "Regression: diagnostics, GLMs, quantiles, regularization, curvature",
            "Error analysis and advanced statistics",
        ],
    ),
    _m(
        "scientist",
        "💡",
        "Model Explainer",
        "pages/25_Model_Explainer.py",
        "Understand and audit a trained model.",
        [
            "Permutation importance, PDP/ICE, interactions",
            "Local explanations, what-if, and counterfactuals",
            "Fairness report and drift (PSI)",
        ],
    ),
    _m(
        "scientist",
        "📒",
        "MLOps",
        "pages/26_MLOps.py",
        "Track, version, serve, and monitor models.",
        [
            "Run tracking, comparison, and reproduction code",
            "Versioned registry with stages and model cards",
            "FastAPI template, batch scoring checks, drift, live monitoring",
        ],
    ),
    _m(
        "shared",
        "🤖",
        "AI Analyst",
        "pages/5_AI_Analyst.py",
        "Ask a question; a planner runs approved analytical tools.",
        [
            "Quick answers or multi-step investigations",
            "Whitelisted tools only, with an audit log",
            "Claims checked against computed evidence",
            "Gemini, Claude, Ollama, or fully offline",
        ],
    ),
    _m(
        "shared",
        "📑",
        "Intelligence Reports",
        "pages/7_Intelligence_Reports.py",
        "Turn verified evidence into a shareable dossier.",
        [
            "Executive summary and methodology log",
            "Includes statistics and model results",
            "Export to HTML or PDF",
        ],
    ),
    _m(
        "workspace",
        "💾",
        "Projects & Sessions",
        "pages/8_Projects_Sessions.py",
        "Save and reopen your work.",
        ["Versioned projects", "Cleaning steps replayed on reopen", "Run history"],
    ),
    _m(
        "workspace",
        "⚙️",
        "Settings",
        "pages/19_Settings.py",
        "Control the AI provider and privacy.",
        [
            "Provider, model, and key (session only)",
            "PII redaction before prompts leave the app",
            "Sensitive-column scan and masking",
            "Tool audit log",
        ],
    ),
)

WORKFLOW: tuple[WorkflowStep, ...] = (
    WorkflowStep(
        1,
        "📥",
        "Ingest",
        "Attach a CSV or Excel file, or pull from SQL, a URL, or a Google Sheet.",
        "pages/1_Executive_Overview.py",
    ),
    WorkflowStep(
        2,
        "🔍",
        "Audit & clean",
        "Review quality issues, preview fixes, and apply them.",
        "pages/2_Data_Quality.py",
    ),
    WorkflowStep(
        3,
        "🧱",
        "Shape",
        "Join, pivot, and document your tables.",
        "pages/11_Transformation_Studio.py",
    ),
    WorkflowStep(
        4,
        "🧭",
        "Explore",
        "Find distributions, correlations, and segment differences.",
        "pages/3_Exploration_Studio.py",
    ),
    WorkflowStep(
        5,
        "🧪",
        "Test",
        "Confirm patterns with statistics and run A/B experiments.",
        "pages/4_Statistical_Intelligence.py",
    ),
    WorkflowStep(
        6,
        "🛠️",
        "Engineer",
        "Create features and check for leakage.",
        "pages/13_Feature_Lab.py",
    ),
    WorkflowStep(
        7, "🔮", "Model", "Train, tune, and compare predictive models.", "pages/6_Model_Lab.py"
    ),
    WorkflowStep(8, "🚀", "Deploy", "Export a model and score new data.", "pages/26_MLOps.py"),
    WorkflowStep(9, "🤖", "Ask", "Query the AI Analyst at any point.", "pages/5_AI_Analyst.py"),
    WorkflowStep(
        10,
        "📑",
        "Report",
        "Export an executive dossier or a dashboard.",
        "pages/7_Intelligence_Reports.py",
    ),
)

PRINCIPLES: tuple[tuple[str, str], ...] = (
    (
        "Tool-first",
        "Every number comes from deterministic code (pandas, SciPy, statsmodels, scikit-learn), never from the LLM.",
    ),
    (
        "Evidence-backed",
        "The AI narrates computed evidence, and its claims are validated against it.",
    ),
    ("Auditable", "Cleaning steps, tool calls, and model provenance are logged and reversible."),
    (
        "Private by design",
        "Data stays in your session. An LLM is optional, and offline mode needs no key.",
    ),
)

GUIDE: tuple[tuple[str, str], ...] = (
    (
        "Which files can I upload?",
        "CSV, Excel (.xlsx/.xls), and Parquet (on the Data Quality page). Large files are limited by "
        "`MAX_UPLOAD_SIZE_MB`. Multi-sheet workbooks offer a worksheet picker.",
    ),
    (
        "Do I need an API key?",
        "No. All statistics, charts, and models run locally. Set `GEMINI_API_KEY` to add natural-language "
        "interpretation, or enable the mock provider in the AI Analyst for offline demos.",
    ),
    (
        "Is my original data safe?",
        "Yes. Cleaning works on a copy; you can roll back any step or reset to the raw file, and export "
        "the steps as a recipe to replay later.",
    ),
    (
        "How do I trust a result?",
        "Each finding lists its evidence, assumptions, and limitations. The Statistical Intelligence page "
        "shows whether test assumptions held, and the Predictive Lab compares against baselines.",
    ),
    (
        "What should I do first?",
        "Load your data on the Executive Overview, fix high-severity issues in the Data Quality Lab, then "
        "follow the Recommended Next Steps card.",
    ),
)
