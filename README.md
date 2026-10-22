# AI Data Analyst Agent

> **From raw data to validated insights and informed decisions.**

An enterprise-grade, autonomous data analytics platform combining the disciplines of data analysis, statistics, machine learning, and strategic business consulting. 

Deterministic computational engines (Pandas, NumPy, SciPy, statsmodels, scikit-learn) calculate verified metrics, while an LLM narrative layer synthesizes findings into natural language using an evidence-backed interpretation framework.

---

## 🏗️ Repository Architecture

```text
DA_Agent/
├── app.py                            # Main Streamlit workspace landing page
├── pages/
│   ├── 1_Executive_Overview.py       # High-level metrics & quality score synthesis
│   ├── 2_Data_Quality.py             # Ingestion, profiling, anomaly audits, and cleaning workbench
│   ├── 3_Exploration_Studio.py       # Interactive distributions & automated charts
│   ├── 4_Statistical_Intelligence.py # Hypothesis tests, effect sizes, and assumptions
│   ├── 5_AI_Analyst.py               # Conversational workspace & 6-layer interpretation
│   ├── 6_Predictive_Lab.py           # Forecasting, ML models, and walk-forward checks
│   └── 7_Intelligence_Reports.py     # PDF/HTML exportable intelligence dossiers
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py               # Typed Pydantic configuration & secret masking
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exceptions.py             # Domain exception hierarchy (user vs tech errors)
│   │   ├── logging_config.py         # Structured logging with rerun deduplication
│   │   ├── session_state.py          # Session state keys, defaults, and lifecycle
│   │   └── ui_components.py          # Reusable dark-mode UI layout components
│   ├── models/
│   │   ├── __init__.py
│   │   └── common.py                 # Pydantic schemas: Metadata, Steps, Plans, Findings
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py                 # CSV/Excel/Parquet loaders, fallbacks, & ManagedDataset
│   │   ├── models.py                 # ColumnSchema and DatasetIngestionReport schemas
│   │   ├── schema.py                 # Automatic schema inference & semantic type heuristics
│   │   ├── transformation_history.py # Reversible transformation pipeline (undo & reset)
│   │   ├── transformations.py        # Deterministic cleaning operations & dry-run previews
│   │   └── validators.py             # Size bounds, extensions, and content fingerprints
│   ├── profiling/
│   │   ├── __init__.py
│   │   ├── consistency.py            # Whitespace, casing, and domain constraint checks
│   │   ├── duplicates.py             # Exact duplicate and key conflict detectors
│   │   ├── missing_values.py         # Null percentage audits & suspicious placeholder tokens
│   │   ├── models.py                 # QualityIssue, QualityScore, and DatasetProfile schemas
│   │   ├── outliers.py               # IQR, Z-Score, and Modified Z-Score (MAD) detectors
│   │   ├── profiler.py               # Comprehensive dataset profiling orchestrator
│   │   └── quality_score.py          # Transparent multi-dimensional scoring framework
│   ├── analytics/                    # Deterministic statistical engines (Phase 3+)
│   ├── agent/                        # Autonomous planning & tool execution (Phase 4+)
│   ├── interpretation/               # Structured 6-layer narrative generation (Phase 4+)
│   ├── reporting/                    # Dossier compilation & rendering (Phase 6+)
│   └── security/                     # Code guardrails & privacy controls
├── tests/
│   ├── conftest.py                   # Pytest fixtures and environment isolation
│   ├── test_settings.py              # Configuration validation and secret safety tests
│   ├── test_session_state.py         # Session lifecycle & rerun preservation tests
│   ├── test_models.py                # Pydantic schema validation & serialization tests
│   ├── test_ingestion.py             # Ingestion, encodings, delimiters, Excel sheets, Parquet
│   ├── test_schema.py                # Semantic type inference, identifiers, constants, targets
│   ├── test_profiling.py             # Missingness, duplicates, outliers (IQR/MAD), consistency
│   ├── test_quality_score.py         # Quality score formulas, weights, and penalty calculations
│   ├── test_transformations.py       # Dry-run previews, immutable raw data, undo, and reset
│   └── test_app_smoke.py             # Multi-page AppTest execution with active datasets
├── sample_data/
│   ├── README.md
│   └── ecommerce_customer_churn.csv  # Sample dataset with intentional quality anomalies
├── artifacts/                        # Generated reports and analysis artifacts
├── .streamlit/
│   └── config.toml                   # Dark-first theme and server settings
├── .env.example                      # Safe placeholder environment template
├── pyproject.toml                    # Project configuration (Ruff, Pytest)
├── requirements.txt                  # Pinned runtime & test dependencies
└── README.md
```

---

## 🔍 Data Ingestion & Quality Intelligence (Phase 2)

### 1. Ingestion Capabilities & Formats
- **CSV Ingestion:** Automatic encoding fallback (`utf-8`, `utf-8-sig`, `latin1`, `cp1252`, `iso-8859-1`) and delimiter sniffing (`,`, `;`, `\t`, `|`).
- **Excel Ingestion:** `.xlsx` and `.xls` support via `openpyxl`, with automated sheet listing and sheet selection.
- **Parquet Ingestion:** High-speed columnar ingestion via `pyarrow`.
- **Integrity Safeguards:** Enforces configurable file size limits (`MAX_UPLOAD_SIZE_MB`), rejects empty files, detects and disambiguates duplicate column headers, and computes a non-sensitive SHA-256 content fingerprint.

### 2. Schema Inference & Semantic Classification
- Automatically detects semantic column roles:
  - **Numeric:** Quantitative continuous metrics.
  - **Categorical:** Discrete groups and low-cardinality codes.
  - **Datetime:** Native timestamps or text dates with high parse rates.
  - **Boolean:** Native flags and binary 0/1 or yes/no indicators.
  - **Identifier:** Keys, IDs, and UUIDs flagged to prevent statistical misuse.
  - **Constant / Near-Constant:** Columns with zero or trivial variance.
  - **Potential Target Candidates:** Outcome variables for future modeling (`churn`, `price`, `revenue`, `status`).

### 3. Data Quality Scoring Methodology
The quality score ranges from **0 to 100** based on an explicit, weighted multi-dimensional formula:

$$\text{Overall Score} = 0.40 \cdot \text{Completeness} + 0.25 \cdot \text{Uniqueness} + 0.20 \cdot \text{Consistency} + 0.15 \cdot \text{Validity}$$

- **Completeness (40%):** Evaluates non-missing cell coverage: $\max(0, 100 - \text{missing\_cell\_pct} \times 1.5)$.
- **Uniqueness (25%):** Penalizes duplicate rows: $\max(0, 100 - \text{duplicate\_row\_pct} \times 2.0)$.
- **Consistency (20%):** Penalizes formatting anomalies (unstripped whitespace, mixed categorical casings).
- **Validity (15%):** Penalizes severe statistical outliers and domain constraint violations (e.g. negative age/prices).

### 4. Interactive Cleaning Workbench & Immutability
- **Immutable Raw Data:** The original uploaded DataFrame (`_raw_df`) is never modified or mutated.
- **Dry-Run Previews:** Users inspect before/after row counts, cell modification metrics, and sample row comparisons prior to execution.
- **Supported Operations:**
  - `drop_column`: Remove empty or constant columns.
  - `drop_duplicates`: Eliminate duplicate records with configurable subset and keep strategies.
  - `impute_numeric`: Median, mean, or constant imputation.
  - `impute_categorical`: Mode or placeholder value imputation.
  - `strip_whitespace`: Normalize unstripped strings.
  - `standardize_casing`: Harmonize categorical casing to title/lower/upper case.
  - `cap_outliers`: Winsorize extreme values to configurable percentiles.
  - `convert_type`: Safely convert date or numeric strings.
- **Reversible Pipeline:** Replay-based `Undo Last Transformation` and `Reset All to Raw` guarantees full reproducibility without bloated in-memory copies.

---

## 🚀 Quickstart Guide (Pop!_OS / Linux Terminal)

### 1. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 2. Activate the Environment

```bash
source .venv/bin/activate
```

## 📈 Exploratory Data Analysis & Interactive Analytics (Phase 3)

### 1. Multi-Tab Exploration Workspace (`pages/3_Exploration_Studio.py`)
- **Dataset Overview:** Total rows, columns, memory consumption in MB, missing cell percentages, duplicate counts, and identifier safeguards that prevent statistical distortion.
- **Variable Explorer:** Dynamic univariate profiling:
  - *Numeric:* Mean, median, std dev, variance, min/max, quartiles, IQR, skewness, kurtosis, zero/negative counts, and outlier counts via Tukey's fences.
  - *Categorical:* Unique category counts, frequency bar charts, dominant category share, rare categories (<1%), and missing/blank counts.
  - *Datetime:* Span in days, temporal interval analysis, and chronological metric trends.
- **Distribution Diagnostics:** Configurable histogram bins (10–80), Tukey box plots, violin density plots with quartile markers, and empirical quantile breakdowns. Includes actionable central tendency advice (mean vs. median divergence).
- **Correlations & Relationships:** Pearson & Spearman pairwise correlation heatmaps with 2-decimal annotations, top correlated pair rankings with sample size $N$ and p-values, bivariate scatter plots with OLS trendlines, and categorical association via Cramér's V.
- **Grouped Disparity Analysis:** Segment comparisons across categorical dimensions using flexible aggregation functions (`mean`, `median`, `sum`, `count`, `std`), disparity ratios between top and bottom groups, and sample size imbalance warnings.
- **Multivariate Studio:** Interactive scatter plotting with X-axis, Y-axis, categorical color grouping, and quantitative bubble size encoding, supported by client performance sampling safeguards (2,500 record ceiling).
- **Automated Insights & Chart Recommendations:** Fully deterministic scoring of optimal chart types and compilation of grounded 6-layer analytical findings.

### 2. Grounded 6-Layer Interpretation Engine (`src/analytics/interpretation.py`)
Deterministic, calculation-backed analytical findings adhering to the 6-layer interpretation model without LLM hallucination:
1. **Observation:** What did the data reveal?
2. **Quantitative Evidence:** Exact calculated values, units, and sample sizes.
3. **Statistical Context:** Distributional shape, variability, missingness, or descriptive uncertainty (strictly non-causal).
4. **Practical Implication:** Operational and analytical impact.
5. **Limitations & Confounders:** Plausible confounding variables, sampling effects, and alternative explanations.
6. **Next Investigation:** Concrete follow-up analyses with available dataset features.

### 3. Deterministic Chart Recommender (`src/analytics/chart_recommender.py`)
Selects and ranks optimal visualizations based on column cardinality, statistical data types, and analytical objectives, assigning a 0–100 suitability score with documented rationale and limitations.

---

## 🧪 Statistical Intelligence & Inference Engine (Phase 4)

### 1. Multi-Objective Statistical Workspace (`pages/4_Statistical_Intelligence.py`)
- **Two-Group Comparisons:** Welch's t-test (heteroscedastic default), Student's t-test (equal variance), and Mann-Whitney U rank-sum test with exact p-values, Welch-Satterthwaite confidence intervals, and Cohen's d / Hedges' g / Cliff's delta effect sizes.
- **Paired Comparisons:** Paired Student's t-test and Wilcoxon signed-rank test for repeated measures and aligned pre/post observations, with paired confidence intervals and $d_z$ effect sizes.
- **Multi-Group Comparisons:** One-Way ANOVA omnibus testing across 3+ segments with $\eta^2$ and $\omega^2$ effect sizes and between/within variance breakdowns.
- **Categorical Association:** Pearson Chi-Square test of independence, Fisher's exact test for sparse $2 \times 2$ tables, Cochran's expected frequency checks, standardized residual matrices, and Cramér's V.
- **Correlation Inference:** Pearson and Spearman correlation tests with exact p-values, Fisher's $z$-transformation confidence intervals, and non-causal interpretative safeguards.
- **OLS Linear Regression:** Ordinary Least Squares estimation with dummy-encoded categorical predictors, parameter uncertainty (std errors, t-statistics, 95% CIs), $R^2$, adjusted $R^2$, overall model F-tests, and comprehensive diagnostics:
  - *Multicollinearity:* Variance Inflation Factor (VIF).
  - *Heteroscedasticity:* Breusch-Pagan LM test.
  - *Autocorrelation:* Durbin-Watson test.
  - *Residual Normality:* Jarque-Bera test.
  - *Influential Observations:* Cook's distance ($4/N$ threshold).

### 2. Multiplicity Correction & Assumption Diagnostics
- **Multiple-Testing Correction (`src/analytics/multiple_testing.py`):** Benjamini-Hochberg False Discovery Rate (FDR) control, Bonferroni FWER, and Holm-Bonferroni step-down correction.
- **Assumption Verification (`src/analytics/assumption_checks.py`):** Shapiro-Wilk and D'Agostino-Pearson normality tests, Levene's / Brown-Forsythe homoscedasticity test, and normal Q-Q plot generation.
- **Robustness & Sensitivity (`src/analytics/robustness.py`):** Compares baseline findings against non-parametric alternatives, outlier-trimmed samples (Tukey 1.5x IQR), and conservative significance levels ($\alpha = 0.01$).

### 3. Grounded 8-Layer Interpretation Framework (`src/analytics/statistical_interpretation.py`)
Deterministic, calculation-backed narrative generated for every inferential result:
1. **Research Question:** Specific empirical relationship or difference under evaluation.
2. **Method & Rationale:** Why the test was selected and its mathematical justification.
3. **Quantitative Evidence:** Actual test statistics, degrees of freedom, sample sizes, and p-values.
4. **Statistical Evidence:** Relation to the null hypothesis without confusing statistical detectability with proof.
5. **Effect Size & Practical Importance:** Standardized magnitude (Cohen's d, Cliff's delta, Cramér's V, $\eta^2$, $R^2$) evaluated in operational context.
6. **Assumptions & Limitations:** Diagnostic warnings, dependence concerns, sample limitations, and omitted confounders.
7. **Grounded Conclusion:** Humble, non-causal conclusion language.
8. **Recommended Next Investigation:** Concrete follow-up analyses and sensitivity checks.

---

## 🤖 Phase 5: Evidence-Grounded AI Interpretation Engine

Phase 5 connects deterministic data profiling, EDA, and statistical inference to a configurable Large Language Model (LLM).

### Core Philosophy
**The LLM interprets evidence; deterministic Python functions calculate evidence.** Zero hallucination or fabrication of metrics, charts, p-values, or unexecuted actions.

### Key Capabilities
- **Configurable LLM Provider Layer:** Pluggable interface supporting Google Gemini (`google-genai` SDK) and a deterministic offline `MockLLMProvider`. No API keys are leaked in logs or exceptions.
- **Evidence Pack (Source of Truth):** Assembles scoped, verified analytical context with stable IDs (`ev_...`), exact metric values, sample sizes, and provenance.
- **Numerical Claim Validation:** Deterministically verifies that all quantitative statements made in analyst findings correspond directly to active Evidence Pack items within strict floating-point tolerances.
- **Contradiction Detection:** Proactively flags correlation-to-causation fallacies, directional conflicts (claiming increase when trend is negative), false significance assertions ($p \ge 0.05$), and stale dataset references.
- **Controlled Analytical Tool Registry:** Safe execution wrappers (`get_dataset_overview`, `calculate_descriptive_statistics`, `compare_groups`, `calculate_correlation`, `fit_regression`, `get_regression_diagnostics`, `inspect_missingness`) with audit logs. Arbitrary Python/shell execution is strictly blocked.
- **8-Layer Deep Interpretation Framework:** Generates comprehensive analyses addressing what happened, magnitude, concentration, mechanisms, confounding, robustness, and next actions.
- **Exploratory Hypothesis Generation:** Formulates structured, testable hypotheses with explicit supporting/weakening criteria and exploratory labeling.
- **Conversational Memory:** Preserves multi-turn session context with dataset version scoping, reference resolution, and invalidation upon dataset changes.
- **No-Provider Fallback:** Full application remains functional without external credentials, providing deterministic analytical synthesis.

---

## 🕵️ Phase 6: Agentic Orchestration & Investigation Engine

Phase 6 connects deterministic analytical capabilities and evidence-grounded AI into a coordinated, multi-step investigation engine.

### Core Philosophy
**The LLM proposes plans and synthesizes findings; deterministic Python services execute calculations, enforce DAG dependencies, and validate outputs.**

### Key Capabilities
- **Typed Lifecycle Architecture:** Strict Pydantic models for `InvestigationPlan`, `InvestigationStep`, `InvestigationState`, and `InvestigationResult`.
- **Deterministic DAG Validation (`WorkflowGraphValidator`):** Verifies step dependency acyclicity via Kahn's algorithm, enforces registered tool schemas, blocks prompt-injection attempts, and validates that all referenced columns exist in the active working dataset.
- **Dependency-Aware Step Execution (`StepExecutor`):** Schedules steps in topological order, halts downstream branches on prerequisite failure, and caches deterministic outputs using composite `(version, tool, args)` keys.
- **Explicit State Machine (`InvestigationStateMachine`):** Implements an audited transition graph across 12 lifecycle states (`CREATED`, `PLANNING`, `PLAN_VALIDATION`, `READY`, `RUNNING`, `REVIEWING`, `REVISING`, `SYNTHESIZING`, `COMPLETED`, `FAILED`, `CANCELLED`, `BUDGET_EXCEEDED`).
- **Adaptive Plan Revision (`PlanRevisionEngine`):** Dynamically adjusts remaining steps based on intermediate findings (e.g., automatically appending a robustness check when a significant segment disparity $p < 0.05$ is detected).
- **Deterministic Stopping Criteria & Budgets (`TerminationEvaluator`):** Enforces bounds on steps, tool calls, elapsed time, and plan revisions. Detects and halts infinite loop cycles via plan signature hashing.
- **Error Classification & Bounded Recovery (`ErrorRecoveryClassifier`):** Categorizes failures into transient infrastructure, argument errors, assumption violations, and security breaches.
- **Dual-Mode UI Workspace:** Toggle between **Quick Answer** (instant single-turn synthesis) and **Deep Multi-Step Investigation** (plan preview, user approval, live execution trace, budget indicators, and final investigation report).

---

### 3. Configure Credentials (Optional)

Copy `.env.example` to `.env` and configure your Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY="your-gemini-api-key"
GEMINI_MODEL=gemini-1.5-pro
```
*Note: If no API key is configured, the application runs seamlessly in **Deterministic Offline Mode** or with the built-in **Mock Provider**.*

---

## 🔮 Predictive Analytics Lab & Intelligence Reporting Engine (Phase 7)

### 1. Leakage-Safe Predictive Modeling (`pages/6_Predictive_Lab.py`)
- **Task Inference & Target Selection:** Automated classification of problem category (`Regression`, `Binary Classification`, `Multiclass Classification`, `Time-Series Forecasting`) with suitability verification and ID screening.
- **Leakage-Safe Preprocessing (`src/predictive/preprocessing.py`):** Scikit-learn column transformers fitted exclusively on training data partitions (`SimpleImputer`, `StandardScaler`, `OneHotEncoder(handle_unknown="ignore")`).
- **Splitting Strategies (`src/predictive/split_strategy.py`):** Reproducible stratified partitions, group-aware splitting, and chronological splitting (zero future leakage for ordered series).
- **Benchmark Baselines (`src/predictive/baselines.py`):** Candidate models compared against defensible naive baselines (`DummyRegressor(strategy="mean")`, `DummyClassifier(strategy="most_frequent")`, `NaiveForecaster`, `SeasonalNaiveForecaster`).
- **Candidate Models:** Ridge, Linear Regression, Random Forest, HistGradientBoosting, and Holt-Winters Exponential Smoothing.
- **Holdout Diagnostics & Curves (`src/predictive/diagnostics.py`):** Actual vs. Predicted scatter plots with 45-degree reference, residual plots, Confusion Matrix heatmaps, and ROC curves.
- **Permutation Explainability (`src/predictive/explainability.py`):** Test-set permutation feature importance with feature name preservation and non-causal interpretative warnings.
- **Conformal Uncertainty (`src/predictive/uncertainty.py`):** Distribution-free conformal prediction intervals calculated from test residual empirical quantiles.
- **Model Registry & Provenance (`src/predictive/model_registry.py`):** Tracks dataset version provenance, random seeds, and flags models as stale if the dataset is modified.

### 2. Evidence-Backed Intelligence Reporting (`pages/7_Intelligence_Reports.py`)
- **Evidence Collector (`src/reporting/evidence_collector.py`):** Aggregates verified outputs across Ingestion, Data Quality, EDA, Statistical Inference, AI Investigation, and Predictive Modeling. Every cited metric links to a stable evidence ID with zero hallucination.
- **Report Types:** Executive Analytics Report, Exploratory Data Analysis Report, Statistical Intelligence Report, AI Investigation Report, Predictive Model Evaluation Report, and Comprehensive Dataset Intelligence Report.
- **Multi-Format Export (`src/reporting/export_service.py`):**
  - **Standalone Interactive HTML:** Responsive design with print CSS for executive presentations.
  - **Downloadable PDF:** Clean, professional styling via ReportLab with automatic table pagination and graceful fallbacks.
- **Deterministic Offline Support:** Operates seamlessly with deterministic template synthesis without requiring an active LLM provider.

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload data in **2. Data Quality Lab**, explore in **3. Exploration Studio**, test inferences in **4. Statistical Intelligence**, orchestrate investigations in **5. AI Analyst Workspace**, train models in **6. Predictive Lab**, and export deliverables in **7. Intelligence Reports**!

### 6. Run Automated Tests

```bash
pytest
```

### 7. Run Code Quality & Formatting Checks

```bash
ruff check .
ruff format --check .
```

---

## 🗺️ Roadmap & Phase Status

| Phase | Milestone | Status |
| :--- | :--- | :---: |
| **Phase 1** | **Project Architecture, Shell, Config, Session State & Tests** | **Complete** ✅ |
| **Phase 2** | **Data Ingestion, Profiling, Quality Scoring & Cleaning Workbench** | **Complete** ✅ |
| **Phase 3** | **Exploratory Data Analysis (EDA) & Interactive Analytics Engine** | **Complete** ✅ |
| **Phase 4** | **Statistical Intelligence, Hypothesis Testing & Inference Engine** | **Complete** ✅ |
| **Phase 5** | **Evidence-Grounded AI Interpretation Engine & Claim Validation** | **Complete** ✅ |
| **Phase 6** | **Agentic Multi-Step Investigation Orchestration & Plan Validation** | **Complete** ✅ |
| **Phase 7** | **Predictive Intelligence Lab & Executive Report Export Engine** | **Complete** ✅ |
| **Phase 8** | Final Production Hardening, Performance Optimization & Polish | *Upcoming* |
