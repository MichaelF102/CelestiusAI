# AI Data Analyst Agent

> **From raw data to validated insights and informed decisions.**

An enterprise-grade, autonomous data analytics platform that combines the capabilities of a data analyst, statistician, machine learning engineer, and business consultant. 

Rather than relying on ungrounded LLM hallucinations or rigid dashboards, the system couples deterministic computational engines (Pandas, SciPy, statsmodels, scikit-learn) with an agentic planner and an evidence-backed interpretation engine.

---

## 🏗️ Repository Architecture

```text
DA_Agent/
├── app.py                            # Main Streamlit workspace entry point
├── pages/
│   ├── 1_Executive_Overview.py       # High-level metrics & executive summaries
│   ├── 2_Data_Quality.py             # Schema, missingness, and outlier profiling
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
│   ├── ingestion/                    # Dataset parsing & validation (Phase 2+)
│   ├── profiling/                    # Quality scoring & anomaly detection (Phase 2+)
│   ├── analytics/                    # Deterministic statistical engines (Phase 2+)
│   ├── agent/                        # Autonomous planning & tool execution (Phase 4+)
│   ├── interpretation/               # Structured 6-layer narrative generation (Phase 3+)
│   ├── reporting/                    # Dossier compilation & rendering (Phase 6+)
│   └── security/                     # Code guardrails & privacy controls
├── tests/
│   ├── conftest.py                   # Pytest fixtures and environment isolation
│   ├── test_settings.py              # Configuration validation and secret safety tests
│   ├── test_session_state.py         # Session lifecycle & rerun preservation tests
│   ├── test_models.py                # Pydantic schema validation & serialization tests
│   └── test_app_smoke.py             # End-to-end AST syntax and Streamlit AppTest suite
├── sample_data/                      # Example datasets for local testing
├── artifacts/                        # Generated reports and analysis artifacts
├── .streamlit/
│   └── config.toml                   # Dark-first theme and server settings
├── .env.example                      # Safe placeholder environment template
├── pyproject.toml                    # Project configuration (Ruff, Pytest)
├── requirements.txt                  # Minimal pinned runtime & test dependencies
└── README.md
```

---

## 📐 The 6-Layer Interpretation Framework

Every finding delivered by the agent adheres to a strict six-layer verification framework:

```
[1. Observation]             What anomaly or pattern was detected?
       ↓
[2. Quantitative Evidence]   Measured figures, baselines, and percentage variances
       ↓
[3. Statistical Confidence]  Sample size checks, p-values, confidence intervals, limitations
       ↓
[4. Explanatory Breakdown]   Decomposition of primary drivers vs. secondary associations
       ↓
[5. Practical Implication]   Operational consequences and business significance
       ↓
[6. Recommended Action]      Targeted next queries or empirical checks to reduce uncertainty
```

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

### 4. Configure Local Environment (Optional)

```bash
cp .env.example .env
```

*Note: In Phase 1, the application operates in local mode and does not require an external Gemini API key.*

### 5. Run the Application

```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

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

## 🔒 Security & Privacy Guardrails

- **Zero Secret Exposure:** API keys and credentials are never printed in logs, UI diagnostics, or serialization representations.
- **Strict Separation of Concerns:** Computational metrics are computed strictly by deterministic Python libraries (`Pandas`, `SciPy`, `statsmodels`), preventing LLM hallucinations of statistical figures.
- **Rerun-Safe Session State:** Streamlit session state initialization protects existing data structures against accidental resets on script reruns.

---

## 🗺️ Roadmap & Phase Status

| Phase | Milestone | Status |
| :--- | :--- | :--- |
| **Phase 1** | **Project Architecture, Shell, Config, Session State & Tests** | **Complete** ✅ |
| **Phase 2** | Data Ingestion, Profiling, EDA & Statistical Tools | *Upcoming* |
| **Phase 3** | Structured Evidence & Verification Engine | *Upcoming* |
| **Phase 4** | Autonomous Agent Planning & Tool Orchestration | *Upcoming* |
| **Phase 5** | Time-Series Forecasting & Predictive Modeling Lab | *Upcoming* |
| **Phase 6** | Executive Reports & Production Hardening | *Upcoming* |
