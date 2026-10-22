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

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open `http://localhost:8501`, navigate to **2. Data Quality Lab**, and test with `sample_data/ecommerce_customer_churn.csv`!

### 5. Run Automated Tests

```bash
pytest
```

### 6. Run Code Quality & Formatting Checks

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
| **Phase 3** | Exploratory Data Analysis (EDA) Studio & Statistical Intelligence | *Upcoming* |
| **Phase 4** | Autonomous Agent Planning & 6-Layer Interpretation Engine | *Upcoming* |
| **Phase 5** | Time-Series Forecasting & Predictive Modeling Lab | *Upcoming* |
| **Phase 6** | Executive Reports & Production Hardening | *Upcoming* |
