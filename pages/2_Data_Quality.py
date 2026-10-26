"""Page 2: Data Quality Lab - Ingestion, profiling, issue audit, and interactive cleaning workbench."""

import json
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st
from src.config.settings import get_settings
from src.core.code_dq import (
    IMPUTE_STRATEGIES,
    OUTLIER_TREATMENTS,
    consistency_snippet,
    convert_types_snippet,
    derive_rules,
    imputation_snippet,
    quality_score_snippet,
    schema_snippet,
    validation_snippet,
)
from src.core.code_r_sql import (
    r_duplicates,
    r_missing,
    r_outliers,
    r_profile,
    sql_duplicates,
    sql_profile,
)
from src.core.code_snippets import (
    cleaning_snippet,
    duplicates_snippet,
    load_kwargs,
    missing_snippet,
    outliers_snippet,
    profile_snippet,
)
from src.core.exceptions import DataAnalystError
from src.core.logging_config import get_logger
from src.core.session_state import (
    KEY_BASELINE_QUALITY_SCORE,
    KEY_CURRENT_DATASET,
    KEY_DATA_QUALITY_REPORT,
    get_transformation_pipeline,
    init_session_state,
    update_quality_profile,
)
from src.core.ui_components import (
    esc,
    render_code_button,
    render_code_export,
    render_empty_state_banner,
    render_page_header,
    render_sidebar_status,
    severity_chip,
)
from src.ingestion.loader import ManagedDataset, get_excel_sheet_names
from src.ingestion.recipe import export_recipe, replay_recipe
from src.ingestion.service import ingest_into_state, needs_ingestion
from src.ingestion.transformation_history import TransformationPipeline
from src.ingestion.transformations import preview_transformation
from src.profiling.actions import (
    actionable_issues,
    apply_issues,
    is_safe_issue,
    reprofile,
    spec_from_issue,
)
from src.profiling.certificate import build_quality_certificate
from src.profiling.duplicates import find_candidate_primary_keys
from src.profiling.missing_values import (
    compute_missingness_correlation,
    compute_nullity_matrix,
)
from src.profiling.outliers import (
    detect_outliers_iqr,
    detect_outliers_modified_zscore,
    detect_outliers_zscore,
)

logger = get_logger("ui.data_quality")
settings = get_settings()

st.set_page_config(
    page_title=f"{settings.app_name} | Data Quality Lab",
    page_icon="🔍",
    layout="wide",
)

init_session_state()
render_sidebar_status()

render_page_header(
    title="🔍 Data Quality Lab",
    subtitle="Ingest raw data, audit structural integrity, identify anomalies, and apply verified transformations.",
)

# Initialize TransformationPipeline in session state if needed
pipeline: TransformationPipeline = get_transformation_pipeline(st.session_state)

SECTIONS = ["📊 Overview", "🧬 Schema & Columns", "🔎 Issue Explorer", "🛠️ Clean"]
SEVERITIES = ("critical", "high", "medium", "low")


def _go_fix(issue_id: str) -> None:
    """Callback: jump to the Clean section with this issue preselected."""
    st.session_state["dq_section"] = SECTIONS[3]
    st.session_state["dq_selected_issue_id"] = issue_id


# ---------------------------------------------------------
# UPLOAD (compact)
# ---------------------------------------------------------
with st.expander(
    "📥 Upload or replace dataset",
    expanded=st.session_state.get(KEY_CURRENT_DATASET) is None,
):
    st.caption(
        f"Supported formats: **CSV, Excel (.xlsx, .xls), Parquet**. Maximum configured file size: **{settings.max_upload_size_mb} MB**."
    )

    u_col1, u_col2 = st.columns([3, 2])

    with u_col1:
        uploaded_file = st.file_uploader(
            "Select or drop dataset file",
            type=["csv", "xlsx", "xls", "parquet"],
            help="Original raw records are kept immutable. Transformations run on a working copy.",
            key="dataset_file_uploader",
        )

    sheet_selected: str | None = None
    if uploaded_file and uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        with u_col2:
            try:
                sheets = get_excel_sheet_names(uploaded_file.getvalue())
                if len(sheets) > 1:
                    sheet_selected = st.selectbox("Select Excel Worksheet", sheets)
                elif sheets:
                    sheet_selected = sheets[0]
                    st.caption(f"Default Sheet: `{sheet_selected}`")
            except Exception as e:
                st.warning(f"Could not inspect Excel sheets: {e}")

    if uploaded_file is not None:
        if needs_ingestion(st.session_state, uploaded_file.name, sheet_selected):
            with st.spinner(f"Ingesting and profiling '{uploaded_file.name}'..."):
                try:
                    managed = ingest_into_state(
                        st.session_state,
                        file_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                        max_size_mb=settings.max_upload_size_mb,
                        sheet_name=sheet_selected,
                    )
                    st.success(
                        f"Successfully loaded `{uploaded_file.name}` "
                        f"({len(managed.working_df):,} rows, {len(managed.working_df.columns):,} cols)!"
                    )
                    st.rerun()
                except DataAnalystError as dae:
                    st.error(f"**Ingestion Error:** {dae.user_message}")
                    logger.error(f"User-facing ingestion failure: {dae.message}")
                except Exception as ex:
                    st.error(f"**Unexpected Error:** Failed to read file: {ex}")
                    logger.exception("Unexpected ingestion error")

dataset: ManagedDataset | None = st.session_state.get(KEY_CURRENT_DATASET)
profile = st.session_state.get(KEY_DATA_QUALITY_REPORT)

if dataset is None or profile is None:
    render_empty_state_banner("Data Quality Lab")
    st.stop()

# ---------------------------------------------------------
# STATUS STRIP (visible from every section)
# ---------------------------------------------------------
qs = profile.quality_score
baseline = st.session_state.get(KEY_BASELINE_QUALITY_SCORE)
severity_counts = Counter(i.severity for i in profile.quality_issues)

s1, s2, s3, s4, s5 = st.columns(5)
with s1:
    with st.container(border=True):
        delta = (
            f"{qs.overall_score - baseline:+.0f} vs. raw"
            if baseline is not None and pipeline.records
            else None
        )
        st.metric("Quality Score", f"{qs.overall_score:.0f}/100", delta=delta)
with s2:
    with st.container(border=True):
        st.metric("Issues", f"{len(profile.quality_issues)}")
        st.caption(
            " · ".join(f"{severity_counts[s]} {s}" for s in SEVERITIES if severity_counts[s])
            or "None"
        )
with s3:
    with st.container(border=True):
        st.metric("Missing Cells", f"{profile.missing_cell_percentage:.1f}%")
with s4:
    with st.container(border=True):
        st.metric("Duplicate Rows", f"{profile.duplicate_row_percentage:.1f}%")
with s5:
    with st.container(border=True):
        st.metric("Transformations", f"{len(pipeline.records)}")

cert_json = json.dumps(
    build_quality_certificate(
        dataset.filename,
        dataset.fingerprint,
        profile,
        len(dataset.working_df),
        len(dataset.working_df.columns),
        len(pipeline.records),
        baseline,
    ),
    indent=2,
)
nav_col, export_col = st.columns([5, 1])
with nav_col:
    st.session_state.setdefault("dq_section", SECTIONS[0])
    section = st.segmented_control(
        "Section", SECTIONS, key="dq_section", label_visibility="collapsed"
    )
    section = section or SECTIONS[0]
LOAD = load_kwargs(dataset)
_df_now = dataset.working_df
_missing_cols = [c for c in _df_now.columns if _df_now[c].isna().any()]
_CONVERSIONS = {
    "datetime64[ns]": "datetime",
    "float64": "numeric",
    "int64": "numeric",
    "bool": "boolean",
    "category": "category",
}
_conv_map = {
    c: _CONVERSIONS.get(s.proposed_conversion, "string")
    for c, s in dataset.schema_info.items()
    if s.proposed_conversion
}

with st.container():
    code_cols = st.columns(5)
    if section == SECTIONS[0]:
        with code_cols[0]:
            render_code_button(
                "Dataset profile",
                lambda o: profile_snippet(options=o, **LOAD),
                key="code_dq_profile",
                r_builder=lambda: r_profile(dataset.filename),
                sql_builder=lambda: sql_profile("data", list(_df_now.columns)),
            )
        with code_cols[1]:
            render_code_button(
                "Quality score",
                lambda o: quality_score_snippet(options=o, **LOAD),
                key="code_dq_score",
            )
    elif section == SECTIONS[1]:
        with code_cols[0]:
            render_code_button(
                "Schema report", lambda o: schema_snippet(options=o, **LOAD), key="code_dq_schema"
            )
        with code_cols[1]:
            render_code_button(
                "Text consistency",
                lambda o: consistency_snippet(options=o, **LOAD),
                key="code_dq_consist",
            )
        with code_cols[2]:
            render_code_button(
                "Type conversion",
                lambda o: convert_types_snippet(
                    _conv_map or dict.fromkeys(_df_now.columns[:1], "string"), options=o, **LOAD
                ),
                key="code_dq_types",
            )
        with code_cols[3]:
            render_code_button(
                "Validation rules (pandera)",
                lambda o: validation_snippet(derive_rules(_df_now), "pandera", options=o, **LOAD),
                key="code_dq_pandera",
            )
        with code_cols[4]:
            render_code_button(
                "Validation rules (asserts)",
                lambda o: validation_snippet(derive_rules(_df_now), "asserts", options=o, **LOAD),
                key="code_dq_asserts",
            )
    elif section == SECTIONS[3]:
        with code_cols[0]:
            render_code_button(
                "Cleaning pipeline",
                lambda o: cleaning_snippet(pipeline.specs, options=o, **LOAD),
                key="code_dq_clean",
            )
        with code_cols[1]:
            strategy = st.selectbox(
                "Imputation",
                IMPUTE_STRATEGIES,
                key="dq_impute_strategy",
                label_visibility="collapsed",
            )
        with code_cols[2]:
            render_code_button(
                f"Imputation ({strategy})",
                lambda o: imputation_snippet(
                    _missing_cols or list(_df_now.columns[:1]), strategy, options=o, **LOAD
                ),
                key="code_dq_impute",
            )
        with code_cols[3]:
            render_code_button(
                "Validation after cleaning",
                lambda o: validation_snippet(derive_rules(_df_now), "asserts", options=o, **LOAD),
                key="code_dq_validate_clean",
            )
with export_col:
    st.download_button(
        "📜 Audit certificate",
        data=cert_json,
        file_name=f"quality_audit_{dataset.dataset_id}.json",
        mime="application/json",
        key="btn_download_quality_cert",
        use_container_width=True,
    )

# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------
if section == SECTIONS[0]:
    st.subheader("Top Issues")
    severity_filter = st.multiselect(
        "Filter issues by severity:",
        options=list(SEVERITIES),
        default=["critical", "high", "medium"],
    )
    ranked = sorted(
        (i for i in profile.quality_issues if i.severity in severity_filter),
        key=lambda i: SEVERITIES.index(i.severity),
    )
    if not ranked:
        st.success("No issues detected for the selected severity filters!")
    for issue in ranked[:5]:
        with st.container(border=True):
            st.markdown(
                f"{severity_chip(issue.severity)} **{esc(issue.title)}**<br>"
                f"<span class='fcard-body'>{esc(issue.measured_evidence)}</span><br>"
                f"<span class='fcard-body'>➡️ {esc(issue.recommended_action)}</span>",
                unsafe_allow_html=True,
            )
            if issue.suggested_action_type is not None and not issue.is_resolved:
                st.button(
                    "🛠️ Fix this",
                    key=f"fix_{issue.issue_id}",
                    on_click=_go_fix,
                    args=(issue.issue_id,),
                )
            else:
                st.caption("Manual review recommended.")
    if len(ranked) > 5:
        st.caption(f"Showing the top 5 of {len(ranked)}. Full list below.")
    if ranked:
        with st.expander(f"All {len(ranked)} issues (table)"):
            st.dataframe(
                [
                    {
                        "Severity": i.severity.upper(),
                        "Category": i.category.upper(),
                        "Title": i.title,
                        "Affected Columns": ", ".join(i.affected_columns),
                        "Affected Rows": f"{i.affected_row_count:,}",
                        "Evidence": i.measured_evidence,
                        "Recommendation": i.recommended_action,
                    }
                    for i in ranked
                ],
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("Score Breakdown")
    sc_col2, sc_col3, sc_col4, sc_col5 = st.columns(4)
    sc_col2.metric("Completeness (40%)", f"{qs.completeness_score:.1f}%")
    sc_col3.metric("Uniqueness (25%)", f"{qs.uniqueness_score:.1f}%")
    sc_col4.metric("Consistency (20%)", f"{qs.consistency_score:.1f}%")
    sc_col5.metric("Validity (15%)", f"{qs.validity_score:.1f}%")
    st.caption(f"ℹ️ **Scoring rationale:** {qs.explanation}")

    st.markdown("#### Column Completeness")
    col_completeness = [
        {"Column": col, "Completeness (%)": round(100.0 - s.null_percentage, 1)}
        for col, s in profile.column_schemas.items()
    ]
    if col_completeness:
        fig = px.bar(
            col_completeness,
            x="Column",
            y="Completeness (%)",
            color="Completeness (%)",
            color_continuous_scale="Blues",
            range_y=[0, 105],
            template="plotly_dark",
        )
        fig.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20}, height=300)
        st.plotly_chart(fig, use_container_width=True, key="dq_fig_completeness")

# ---------------------------------------------------------
# SCHEMA & COLUMNS
# ---------------------------------------------------------
elif section == SECTIONS[1]:
    st.subheader("Inferred Schema & Semantic Classification")
    st.caption("Auto-detected data types, cardinality, identifier roles, and target candidates.")

    schema_records = []
    for _col, s in dataset.schema_info.items():
        flags = []
        if s.is_identifier:
            flags.append("🔑 Identifier")
        if s.is_potential_target:
            flags.append("🎯 Target")
        if s.is_constant:
            flags.append("⚠️ Constant")
        elif s.is_near_constant:
            flags.append("⚡ Near-Constant")
        if s.is_datetime_candidate and s.inferred_type != "datetime":
            flags.append("📅 Date Candidate")

        schema_records.append(
            {
                "Position": s.position,
                "Column Name": s.column_name,
                "Inferred Type": s.inferred_type.upper(),
                "Pandas Dtype": s.detected_pandas_dtype,
                "Nulls (%)": f"{s.null_count:,} ({s.null_percentage:.1f}%)",
                "Distinct": f"{s.distinct_count:,}",
                "Role / Flags": ", ".join(flags) if flags else "—",
                "Confidence": f"{int(s.type_confidence * 100)}%",
                "Rationale": s.rationale,
                "Proposed Conversion": s.proposed_conversion or "None",
            }
        )

    f_col1, f_col2, f_col3 = st.columns([2, 2, 1])
    name_query = f_col1.text_input(
        "Search columns", placeholder="Type to filter…", key="dq_schema_search"
    )
    type_filter = f_col2.multiselect(
        "Type", sorted({r["Inferred Type"] for r in schema_records}), key="dq_schema_types"
    )
    issue_cols = {c for i in profile.quality_issues for c in i.affected_columns}
    only_issues = f_col3.checkbox("Has issues", key="dq_schema_only_issues")
    visible_records = [
        r
        for r in schema_records
        if (not name_query or name_query.lower() in r["Column Name"].lower())
        and (not type_filter or r["Inferred Type"] in type_filter)
        and (not only_issues or r["Column Name"] in issue_cols)
    ]
    st.dataframe(visible_records, use_container_width=True)
    st.caption(f"Showing {len(visible_records)} of {len(schema_records)} columns.")

    with st.expander("👀 Data preview (first 20 rows)"):
        st.dataframe(dataset.working_df.head(20), use_container_width=True)

    # Column drill-down
    st.markdown("---")
    st.markdown("#### 🔬 Column Drill-Down")
    drill_col = st.selectbox("Inspect column", list(dataset.schema_info.keys()), key="dq_drill_col")
    d_schema = dataset.schema_info[drill_col]
    d_series = dataset.working_df[drill_col]
    dd1, dd2, dd3, dd4 = st.columns(4)
    dd1.metric("Type", d_schema.inferred_type.upper())
    dd2.metric("Missing", f"{d_schema.null_percentage:.1f}%")
    dd3.metric("Distinct", f"{d_schema.distinct_count:,}")
    dd4.metric("Issues", sum(drill_col in i.affected_columns for i in profile.quality_issues))
    if pd.api.types.is_numeric_dtype(d_series):
        d_fig = px.histogram(d_series.dropna(), nbins=30, template="plotly_dark")
        d_fig.update_layout(
            showlegend=False, height=260, margin={"l": 20, "r": 20, "t": 20, "b": 20}
        )
        st.plotly_chart(d_fig, use_container_width=True, key="dq_drill_hist")
    else:
        top_vals = d_series.astype(str).value_counts().head(12).reset_index()
        top_vals.columns = ["Value", "Count"]
        d_fig = px.bar(top_vals, x="Value", y="Count", template="plotly_dark")
        d_fig.update_layout(height=260, margin={"l": 20, "r": 20, "t": 20, "b": 20})
        st.plotly_chart(d_fig, use_container_width=True, key="dq_drill_bar")
    for d_issue in [i for i in profile.quality_issues if drill_col in i.affected_columns]:
        st.markdown(
            f"{severity_chip(d_issue.severity)} **{esc(d_issue.title)}** — "
            f"<span class='fcard-body'>{esc(d_issue.recommended_action)}</span>",
            unsafe_allow_html=True,
        )
        if d_issue.suggested_action_type is not None and not d_issue.is_resolved:
            st.button(
                "🛠️ Fix this",
                key=f"fix_drill_{d_issue.issue_id}",
                on_click=_go_fix,
                args=(d_issue.issue_id,),
            )

    # Storage & Memory Footprint Breakdown
    st.markdown("---")
    st.markdown("#### 💾 Column Storage & In-Memory Footprint")
    mem_series = dataset.working_df.memory_usage(deep=True)
    col_mems = mem_series.drop(labels=["Index"], errors="ignore")
    total_mem_bytes = int(col_mems.sum())
    total_mem_mb = total_mem_bytes / (1024 * 1024)

    mem_c1, mem_c2, mem_c3 = st.columns(3)
    mem_c1.metric("Total In-Memory Size", f"{total_mem_mb:.2f} MB")
    avg_col_mb = (total_mem_mb / len(col_mems)) if len(col_mems) > 0 else 0.0
    mem_c2.metric("Avg Size per Column", f"{avg_col_mb:.2f} MB")
    if not col_mems.empty:
        heaviest_col = str(col_mems.idxmax())
        heaviest_mb = float(col_mems.max()) / (1024 * 1024)
        mem_c3.metric("Heaviest Column", f"{heaviest_col} ({heaviest_mb:.2f} MB)")

    mem_records = [
        {
            "Column": str(c),
            "Memory (KB)": round(float(col_mems[c]) / 1024.0, 1),
            "Memory (MB)": round(float(col_mems[c]) / (1024.0 * 1024.0), 2),
            "Share (%)": round(
                (float(col_mems[c]) / total_mem_bytes) * 100.0 if total_mem_bytes > 0 else 0.0,
                1,
            ),
            "Dtype": str(dataset.working_df[c].dtype),
        }
        for c in col_mems.index
    ]
    mem_records.sort(key=lambda x: x["Memory (KB)"], reverse=True)

    fig_mem = px.bar(
        mem_records,
        x="Column",
        y="Memory (KB)",
        color="Dtype",
        template="plotly_dark",
        title="Column Memory Usage (KB) by Pandas Data Type",
    )
    fig_mem.update_layout(margin={"l": 20, "r": 20, "t": 40, "b": 20}, height=300)
    st.plotly_chart(fig_mem, use_container_width=True, key="dq_fig_mem")

    # Cardinality & Drift Risk Diagnostics
    st.markdown("#### ⚠️ Cardinality & Drift Risk Diagnostics")
    tot_rows = len(dataset.working_df)
    card_risks = []
    for col, s in dataset.schema_info.items():
        card_ratio = s.distinct_count / tot_rows if tot_rows > 0 else 0.0
        if s.is_constant:
            card_risks.append(
                {
                    "Column": col,
                    "Distinct": s.distinct_count,
                    "Ratio": f"{card_ratio:.1%}",
                    "Risk Level": "CRITICAL (Zero Variance)",
                    "Diagnostic Explanation": "Column contains only 1 unique value across all rows; provides zero variance or analytical signal.",
                }
            )
        elif s.inferred_type == "categorical" and card_ratio > 0.40 and not s.is_identifier:
            card_risks.append(
                {
                    "Column": col,
                    "Distinct": s.distinct_count,
                    "Ratio": f"{card_ratio:.1%}",
                    "Risk Level": "HIGH CARDINALITY",
                    "Diagnostic Explanation": f"Over {card_ratio:.0%} unique category labels. May cause severe dimensionality inflation if one-hot encoded.",
                }
            )

    if card_risks:
        st.dataframe(card_risks, use_container_width=True)
    else:
        st.success("No critical cardinality anomalies or zero-variance columns detected.")

# ---------------------------------------------------------
# ISSUE EXPLORER (missingness / duplicates / outliers)
# ---------------------------------------------------------
elif section == SECTIONS[2]:
    ISSUE_KINDS = ["❓ Missingness", "👥 Duplicates", "🎯 Outliers"]
    st.session_state.setdefault("dq_issue_kind", ISSUE_KINDS[0])
    issue_kind = st.segmented_control(
        "Issue type", ISSUE_KINDS, key="dq_issue_kind", label_visibility="collapsed"
    )
    issue_kind = issue_kind or ISSUE_KINDS[0]

    if issue_kind == ISSUE_KINDS[0]:
        st.subheader("Missing-Value Deep Dive")
        render_code_button(
            "Missing values",
            lambda o: missing_snippet(options=o, **LOAD),
            key="code_dq_missing",
            r_builder=lambda: r_missing(dataset.filename),
            sql_builder=lambda: sql_profile("data", list(_df_now.columns)),
        )

        st.write(
            f"Total missing cells across dataset: **{profile.total_missing_cells:,}** "
            f"({profile.missing_cell_percentage:.2f}% of total cells)."
        )

        missing_data = [
            {
                "Column": col,
                "Inferred Type": s.inferred_type,
                "Missing Count": s.null_count,
                "Missing Pct (%)": s.null_percentage,
                "Valid Count": s.non_null_count,
            }
            for col, s in profile.column_schemas.items()
        ]
        missing_data.sort(key=lambda x: x["Missing Count"], reverse=True)

        m_fig = px.bar(
            missing_data,
            x="Column",
            y="Missing Pct (%)",
            color="Missing Pct (%)",
            color_continuous_scale="Reds",
            template="plotly_dark",
            title="Missing Value Percentage by Column",
        )
        m_fig.update_layout(margin={"l": 20, "r": 20, "t": 40, "b": 20}, height=320)
        st.plotly_chart(m_fig, use_container_width=True, key="dq_fig_missing_bar")

        st.dataframe(missing_data, use_container_width=True)

        # Nullity Sparsity Matrix
        show_heavy = st.checkbox(
            "Show nullity map & co-occurrence heatmap",
            value=len(dataset.working_df.columns) <= 25,
            key="dq_show_heavy",
            help="These charts are slower on wide datasets.",
        )
        if show_heavy:
            st.markdown("---")
            st.markdown("#### 🧱 Missingness Sparsity Matrix (Nullity Map)")
            st.caption(
                "Visual representation of missing value positions sampled uniformly across the dataset. Red marks missing cells; Dark Navy marks observed values."
            )
            matrix_df = compute_nullity_matrix(dataset.working_df, max_rows=250)
            if not matrix_df.empty:
                fig_nullity = px.imshow(
                    matrix_df.T,
                    color_continuous_scale=[[0.0, "#ef4444"], [1.0, "#1e293b"]],
                    template="plotly_dark",
                    aspect="auto",
                    labels={
                        "x": "Sampled Records (Uniform Interval)",
                        "y": "Column",
                        "color": "Observed (1=Yes, 0=No)",
                    },
                )
                fig_nullity.update_layout(height=280, margin={"l": 20, "r": 20, "t": 20, "b": 20})
                st.plotly_chart(fig_nullity, use_container_width=True, key="dq_fig_nullity_map")

            # Missingness Co-occurrence / Correlation Heatmap
            corr = compute_missingness_correlation(dataset.working_df)
            if corr is not None and not corr.empty:
                st.markdown("#### 🔗 Missingness Co-occurrence & Correlation Heatmap")
                st.caption(
                    "Measures whether missingness in one column predicts missingness in another (+1 = always missing together, -1 = mutually exclusive)."
                )
                fig_corr = px.imshow(
                    corr,
                    color_continuous_scale="RdBu_r",
                    zmin=-1.0,
                    zmax=1.0,
                    template="plotly_dark",
                    text_auto=".2f",
                )
                fig_corr.update_layout(height=320, margin={"l": 20, "r": 20, "t": 20, "b": 20})
                st.plotly_chart(fig_corr, use_container_width=True, key="dq_fig_missing_corr")

    elif issue_kind == ISSUE_KINDS[1]:
        st.subheader("Duplicate Record Analysis")
        render_code_button(
            "Duplicate analysis",
            lambda o: duplicates_snippet(options=o, **LOAD),
            key="code_dq_dups",
            r_builder=lambda: r_duplicates(dataset.filename),
            sql_builder=lambda: sql_duplicates("data", list(_df_now.columns)),
        )

        dup_count = profile.duplicate_row_count
        dup_pct = profile.duplicate_row_percentage

        d_c1, d_c2 = st.columns(2)
        d_c1.metric("Exact Duplicate Rows", f"{dup_count:,}")
        d_c2.metric("Duplicate Percentage", f"{dup_pct:.2f}%")

        if dup_count > 0:
            st.warning(f"Detected **{dup_count}** exact duplicate rows.")
            dup_rows = dataset.working_df[dataset.working_df.duplicated(keep=False)].head(20)
            st.caption("Sample duplicate rows:")
            st.dataframe(dup_rows, use_container_width=True)
        else:
            st.success("No exact duplicate rows found across the dataset.")

        # Automated Primary Key Discovery
        st.markdown("---")
        st.markdown("#### 🔑 Automated Primary Key & Identifier Discovery")
        st.caption(
            "The engine automatically searches single columns and composite pairs to find columns that uniquely identify each row."
        )
        candidates = find_candidate_primary_keys(dataset.working_df)
        if candidates:
            cand_rows = [
                {
                    "Candidate Key Column(s)": " + ".join(c["columns"]),
                    "Key Structure": c["type"],
                    "Distinct Values": f"{c['distinct_count']:,}",
                    "Uniqueness Rate": f"{c['uniqueness_pct']:.2f}%",
                    "Primary Key Status": "✅ 100% Unique Primary Key"
                    if c["is_100_percent_unique"]
                    else "⚠️ Near-Unique Candidate",
                }
                for c in candidates
            ]
            st.dataframe(cand_rows, use_container_width=True)

            best_key = candidates[0]
            if best_key["is_100_percent_unique"]:
                st.success(
                    f"🎯 **Recommended Entity Key:** `{' + '.join(best_key['columns'])}` "
                    f"is **100.0% unique** across all {len(dataset.working_df):,} rows!"
                )
        else:
            st.info(
                "No individual column or simple composite pair achieves >= 90% uniqueness in this dataset."
            )

        # Key-based duplicate checker
        st.markdown("---")
        st.markdown("#### Custom Entity Key Uniqueness Checker")
        key_cols = st.multiselect(
            "Select business key columns to verify entity uniqueness:",
            options=list(dataset.working_df.columns),
        )
        if key_cols:
            key_dup_mask = dataset.working_df.duplicated(subset=key_cols, keep=False)
            key_dup_count = int(key_dup_mask.sum())
            if key_dup_count > 0:
                st.error(f"Found **{key_dup_count}** rows sharing duplicate keys on `{key_cols}`!")
                st.dataframe(dataset.working_df[key_dup_mask].head(20), use_container_width=True)
            else:
                st.success(f"Composite key `{key_cols}` is 100% unique across all rows.")

    else:
        st.subheader("Outlier & Anomaly Diagnostics")

        numeric_cols = [
            c
            for c, s in profile.column_schemas.items()
            if s.inferred_type == "numeric" and not s.is_identifier and not s.is_constant
        ]

        if not numeric_cols:
            st.info("No quantitative numeric columns available for outlier diagnostics.")
        else:
            o_c1, o_c2, o_c3 = st.columns(3)
            with o_c1:
                selected_num_col = st.selectbox("Select Numeric Column", numeric_cols)
            with o_c2:
                detector_method = st.selectbox(
                    "Detection Method",
                    ["IQR (Tukey's Rule)", "Z-Score", "Modified Z-Score (MAD)"],
                )
            with o_c3:
                if "IQR" in detector_method:
                    multiplier = st.slider("IQR Multiplier (k)", 1.0, 3.0, 1.5, 0.25)
                elif "Modified" in detector_method:
                    multiplier = st.slider("MAD Threshold", 2.0, 5.0, 3.5, 0.5)
                else:
                    multiplier = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.5)

            treatment = st.selectbox(
                "Treatment shown in the code",
                OUTLIER_TREATMENTS,
                key="dq_outlier_treatment",
                help="Adds a before/after comparison to the generated code.",
            )
            _method = (
                "iqr"
                if "IQR" in detector_method
                else ("modified_zscore" if "Modified" in detector_method else "zscore")
            )
            render_code_button(
                f"Outliers in {selected_num_col}",
                lambda o: outliers_snippet(
                    selected_num_col, _method, treatment, multiplier, options=o, **LOAD
                ),
                key="code_dq_outliers",
                r_builder=lambda: r_outliers(dataset.filename, selected_num_col),
            )
            col_series = dataset.working_df[selected_num_col].dropna().astype(float)

            if "IQR" in detector_method:
                res = detect_outliers_iqr(col_series, k=multiplier)
            elif "Modified" in detector_method:
                res = detect_outliers_modified_zscore(col_series, threshold=multiplier)
            else:
                res = detect_outliers_zscore(col_series, threshold=multiplier)

            res_count = int(res["count"])  # type: ignore[arg-type]
            res_pct = float(res["percentage"])  # type: ignore[arg-type]
            lower_b = res.get("lower_bound")
            upper_b = res.get("upper_bound")

            r_col1, r_col2, r_col3 = st.columns(3)
            r_col1.metric("Outlier Count", f"{res_count:,}")
            r_col2.metric("Outlier Percentage", f"{res_pct:.2f}%")
            r_col3.metric("Normal Range", f"[{lower_b}, {upper_b}]")

            # Outlier Distribution Chart
            fig = px.box(
                dataset.working_df,
                y=selected_num_col,
                points="outliers",
                template="plotly_dark",
                title=f"Boxplot Distribution for '{selected_num_col}' with Detected Outliers",
            )
            fig.update_layout(margin={"l": 20, "r": 20, "t": 40, "b": 20}, height=320)
            st.plotly_chart(
                fig, use_container_width=True, key=f"dq_fig_outlier_box_{selected_num_col}"
            )

            # Histogram with Shaded Outlier Rejection Regions
            fig_hist = px.histogram(
                dataset.working_df,
                x=selected_num_col,
                nbins=40,
                template="plotly_dark",
                title=f"Distribution of '{selected_num_col}' with Outlier Rejection Bands",
                color_discrete_sequence=["#38bdf8"],
            )
            if lower_b is not None and upper_b is not None:
                col_min = float(col_series.min())
                col_max = float(col_series.max())
                fig_hist.add_vline(
                    x=lower_b,
                    line_width=2,
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text="Lower Bound",
                )
                fig_hist.add_vline(
                    x=upper_b,
                    line_width=2,
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text="Upper Bound",
                )
                if col_min < lower_b:
                    fig_hist.add_vrect(
                        x0=col_min,
                        x1=lower_b,
                        fillcolor="#ef4444",
                        opacity=0.15,
                        line_width=0,
                        annotation_text="Lower Outliers",
                    )
                if col_max > upper_b:
                    fig_hist.add_vrect(
                        x0=upper_b,
                        x1=col_max,
                        fillcolor="#ef4444",
                        opacity=0.15,
                        line_width=0,
                        annotation_text="Upper Outliers",
                    )

            fig_hist.update_layout(margin={"l": 20, "r": 20, "t": 40, "b": 20}, height=320)
            st.plotly_chart(
                fig_hist, use_container_width=True, key=f"dq_fig_outlier_hist_{selected_num_col}"
            )

            # Distribution Tail & Skewness Insights
            skew_val = float(col_series.skew())
            kurt_val = float(col_series.kurtosis())
            if skew_val > 1.0:
                skew_diag = f"Strong positive right-skew (+{skew_val:.2f}). Most records bunch near lower values; outliers represent extreme high values."
            elif skew_val < -1.0:
                skew_diag = f"Strong negative left-skew ({skew_val:.2f}). Most records bunch near higher values; outliers represent extreme low values."
            else:
                skew_diag = f"Moderate/symmetrical skewness ({skew_val:.2f}). Values roughly balance around the median."

            st.info(f"📊 **Distribution Profile:** {skew_diag} Tail Kurtosis: `{kurt_val:.2f}`.")

            if res_count > 0:
                indices = res.get("outlier_indices", [])  # type: ignore[assignment]
                st.caption(f"Sample outlier records for '{selected_num_col}':")
                st.dataframe(dataset.working_df.loc[indices].head(15), use_container_width=True)

# ---------------------------------------------------------
# CLEAN
# ---------------------------------------------------------
else:
    st.subheader("Cleaning Workbench")
    st.caption(
        "Preview and execute auditable, deterministic transformations. Original data is kept immutable."
    )

    orig_rows = len(dataset.raw_df)
    curr_rows = len(dataset.working_df)
    orig_cols = len(dataset.raw_df.columns)
    curr_cols = len(dataset.working_df.columns)
    comp_c1, comp_c2, comp_c3 = st.columns(3)
    comp_c1.metric(
        "Working Rows",
        f"{curr_rows:,}",
        delta=f"{curr_rows - orig_rows:,}" if curr_rows != orig_rows else None,
    )
    comp_c2.metric(
        "Working Columns",
        f"{curr_cols:,}",
        delta=f"{curr_cols - orig_cols:,}" if curr_cols != orig_cols else None,
    )
    comp_c3.metric("History", f"Step {len(pipeline.records)}")

    def _refresh(message: str) -> None:
        """Re-profile after a change, report the score movement, and rerun."""
        before = profile.quality_score.overall_score
        new_profile = reprofile(dataset)
        update_quality_profile(st.session_state, new_profile, dataset.schema_info)
        st.toast(
            f"{message} · Quality {before:.0f} → {new_profile.quality_score.overall_score:.0f}",
            icon="✅",
        )
        st.rerun()

    st.markdown("---")
    st.markdown("#### 1. Recommended Cleaning Actions")
    actionable = actionable_issues(profile)

    if not actionable:
        st.success("No automated cleaning transformations currently recommended!")
    else:
        safe_issues = [i for i in actionable if is_safe_issue(i)]
        if safe_issues:
            st.markdown("##### ⚡ One-Click Auto-Clean (Safe Non-Destructive Actions)")
            st.caption(
                f"**{len(safe_issues)}** verified safe actions (whitespace trimming, row deduplication, dropping empty columns)."
            )
            if st.button(
                "🚀 Apply All Safe Recommendations", key="btn_auto_clean_safe", type="primary"
            ):
                with st.spinner("Applying safe cleaning transformations..."):
                    applied, errors = apply_issues(pipeline, dataset, safe_issues)
                for err in errors:
                    st.warning(err)
                _refresh(f"Applied {applied} safe transformation(s)")

        issue_labels = {
            i.issue_id: f"[{i.severity.upper()}] {i.title} → {i.recommended_action}"
            for i in actionable
        }
        issue_by_id = {i.issue_id: i for i in actionable}

        with st.expander(f"☑️ Bulk apply ({len(actionable)} available)"):
            bulk_ids = st.multiselect(
                "Select actions to apply together",
                list(issue_labels),
                format_func=issue_labels.get,
                key="dq_bulk_ids",
            )
            if st.button("Apply selected", key="btn_bulk_apply", disabled=not bulk_ids):
                with st.spinner("Applying selected transformations..."):
                    applied, errors = apply_issues(
                        pipeline, dataset, [issue_by_id[i] for i in bulk_ids]
                    )
                for err in errors:
                    st.warning(err)
                _refresh(f"Applied {applied} transformation(s)")

        st.markdown("##### Resolve a single issue")
        ids = list(issue_labels)
        preselected = st.session_state.pop("dq_selected_issue_id", None)
        if preselected in issue_by_id:
            st.session_state["dq_single_issue"] = preselected
        if st.session_state.get("dq_single_issue") not in issue_by_id:
            st.session_state["dq_single_issue"] = ids[0]
        selected_id = st.selectbox(
            "Select issue to resolve:", ids, format_func=issue_labels.get, key="dq_single_issue"
        )
        selected_issue = issue_by_id[selected_id]

        st.write(f"**Explanation:** {selected_issue.explanation}")
        st.write(f"**Consequences:** {selected_issue.potential_consequences}")

        spec_to_preview = spec_from_issue(selected_issue)
        try:
            preview = preview_transformation(dataset.working_df, spec_to_preview)
            st.markdown("##### Dry-Run Preview")
            p_c1, p_c2, p_c3 = st.columns(3)
            p_c1.write(f"**Rows:** {preview.rows_before:,} → **{preview.rows_after:,}**")
            p_c2.write(f"**Cols:** {preview.cols_before:,} → **{preview.cols_after:,}**")
            p_c3.write(f"**Cells Modified:** {preview.affected_cells_count:,}")

            if preview.warnings:
                for w in preview.warnings:
                    st.warning(f"⚠️ {w}")

            # Distribution Shift / Statistical Delta
            target_col = spec_to_preview.column_name
            if target_col and target_col in dataset.working_df.columns:
                s_orig = dataset.working_df[target_col]
                if pd.api.types.is_numeric_dtype(s_orig):
                    st.caption("Distribution delta preview on target column:")
                    delta_rows = [
                        {
                            "Metric": "Valid Count",
                            "Before": f"{int(s_orig.count()):,}",
                            "After Preview": f"{max(0, int(s_orig.count()) - preview.affected_cells_count):,}"
                            if spec_to_preview.action_type in ("drop_column", "drop_duplicates")
                            else f"{int(s_orig.count()):,}",
                        },
                        {
                            "Metric": "Missing Count",
                            "Before": f"{int(s_orig.isna().sum()):,}",
                            "After Preview": "0"
                            if spec_to_preview.action_type == "impute_numeric"
                            else f"{int(s_orig.isna().sum()):,}",
                        },
                        {
                            "Metric": "Mean Value",
                            "Before": f"{float(s_orig.mean()):.2f}",
                            "After Preview": "Imputed / Filtered"
                            if spec_to_preview.action_type in ("impute_numeric", "cap_outliers")
                            else f"{float(s_orig.mean()):.2f}",
                        },
                    ]
                    st.dataframe(delta_rows, use_container_width=True)

            if preview.before_sample:
                st.caption("Before vs. After sample values:")
                b_col, a_col = st.columns(2)
                b_col.dataframe(preview.before_sample)
                a_col.dataframe(preview.after_sample)

            if st.button("✅ Apply Approved Transformation", key="btn_apply_transform"):
                with st.spinner("Applying transformation..."):
                    record = pipeline.apply(spec_to_preview, dataset)
                _refresh(f"Applied: {record.summary}")
        except Exception as e:
            st.error(f"Could not generate preview: {e}")

    # ----- history timeline
    st.markdown("---")
    st.markdown("#### 2. Transformation Timeline")
    if pipeline.records:
        for idx, rec in enumerate(pipeline.records, start=1):
            with st.container(border=True):
                t_info, t_btn = st.columns([5, 1])
                t_info.markdown(
                    f"**Step {idx} · {esc(rec.spec.action_type)}** on "
                    f"`{esc(rec.spec.column_name or 'dataset')}`  \n"
                    f"{esc(rec.summary)}  \n"
                    f"<span class='fcard-body'>Rows {rec.rows_before:,} → {rec.rows_after:,} · "
                    f"{rec.cells_modified:,} cells modified · "
                    f"{rec.applied_at.strftime('%H:%M:%S UTC')}</span>",
                    unsafe_allow_html=True,
                )
                if idx < len(pipeline.records) and t_btn.button(
                    "⏪ Roll back here", key=f"btn_rollback_{idx}"
                ):
                    pipeline.rollback_to(dataset, idx)
                    _refresh(f"Rolled back to step {idx}")

        ctrl_c1, ctrl_c2 = st.columns(2)
        with ctrl_c1:
            if st.button("↩️ Undo Last Transformation", key="btn_undo"):
                undone = pipeline.undo(dataset)
                if undone:
                    _refresh(f"Reverted: {undone.summary}")
        with ctrl_c2:
            if st.button("🔄 Reset All to Original Raw Dataset", key="btn_reset_all"):
                pipeline.reset(dataset)
                _refresh("Restored the original raw dataset")
    else:
        st.caption(
            "No transformations applied yet. Dataset is currently in its original raw state."
        )

    # ----- recipes
    st.markdown("---")
    st.markdown("#### 3. Cleaning Recipe")
    st.caption("Save the applied steps and replay them on another file with the same columns.")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.download_button(
            "💾 Download recipe (JSON)",
            data=export_recipe(pipeline, dataset.filename),
            file_name=f"cleaning_recipe_{dataset.dataset_id}.json",
            mime="application/json",
            disabled=not pipeline.records,
            use_container_width=True,
        )
    with r_col2:
        recipe_file = st.file_uploader("Replay a recipe", type=["json"], key="dq_recipe_upload")
        if recipe_file is not None and st.button("▶️ Apply recipe", key="btn_apply_recipe"):
            try:
                applied, errors = replay_recipe(recipe_file.getvalue(), pipeline, dataset)
            except ValueError as err:
                st.error(str(err))
            else:
                for err in errors:
                    st.warning(err)
                _refresh(f"Replayed {applied} step(s)")

st.markdown("---")
st.page_link("pages/0_Home.py", label="← Return to Home", icon="🏠")

render_code_export()
