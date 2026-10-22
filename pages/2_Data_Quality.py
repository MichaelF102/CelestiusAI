"""Page 2: Data Quality Lab - Ingestion, profiling, issue audit, and interactive cleaning workbench."""

import json

import pandas as pd
import plotly.express as px
import streamlit as st
from src.config.settings import get_settings
from src.core.exceptions import DataAnalystError
from src.core.logging_config import get_logger
from src.core.session_state import (
    KEY_CURRENT_DATASET,
    KEY_DATA_QUALITY_REPORT,
    KEY_DATASET_METADATA,
    KEY_DATASET_SCHEMA,
    KEY_TRANSFORMATION_PIPELINE,
    init_session_state,
    invalidate_downstream_analyses,
    reset_dataset_state,
)
from src.core.ui_components import render_page_header, render_sidebar_status
from src.ingestion.loader import ManagedDataset, get_excel_sheet_names, load_dataset_from_stream
from src.ingestion.transformation_history import TransformationPipeline
from src.ingestion.transformations import preview_transformation
from src.models.common import DatasetMetadata
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
from src.profiling.profiler import profile_dataset

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
    badge="Module 1 • Ingestion & Quality Intelligence",
)

# Initialize TransformationPipeline in session state if needed
if st.session_state.get(KEY_TRANSFORMATION_PIPELINE) is None:
    st.session_state[KEY_TRANSFORMATION_PIPELINE] = TransformationPipeline()
pipeline: TransformationPipeline = st.session_state[KEY_TRANSFORMATION_PIPELINE]

# Navigation tabs
tab_upload, tab_schema, tab_overview, tab_missing, tab_dups, tab_outliers, tab_clean = st.tabs(
    [
        "📥 1. Upload & Preview",
        "📐 2. Schema Explorer",
        "📊 3. Quality Overview",
        "❓ 4. Missingness",
        "👥 5. Duplicates",
        "🎯 6. Outliers",
        "🛠️ 7. Cleaning Workbench",
    ]
)

# ---------------------------------------------------------
# TAB 1: UPLOAD & PREVIEW
# ---------------------------------------------------------
with tab_upload:
    st.subheader("Dataset Ingestion")
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
        # Check if we need to ingest this file
        current_dataset: ManagedDataset | None = st.session_state.get(KEY_CURRENT_DATASET)
        needs_ingestion = (
            current_dataset is None
            or current_dataset.filename != uploaded_file.name
            or (
                sheet_selected
                and current_dataset.ingestion_report.sheet_name_used != sheet_selected
            )
        )

        if needs_ingestion:
            with st.spinner(f"Ingesting and profiling '{uploaded_file.name}'..."):
                try:
                    # Reset previous dataset state to avoid cross-contamination
                    reset_dataset_state(st.session_state)
                    st.session_state[KEY_TRANSFORMATION_PIPELINE] = TransformationPipeline()
                    pipeline = st.session_state[KEY_TRANSFORMATION_PIPELINE]

                    file_bytes = uploaded_file.getvalue()
                    managed = load_dataset_from_stream(
                        file_stream=file_bytes,
                        filename=uploaded_file.name,
                        sheet_name=sheet_selected,
                        max_size_mb=settings.max_upload_size_mb,
                    )

                    # Generate dataset profile
                    profile = profile_dataset(
                        managed.working_df,
                        dataset_id=managed.dataset_id,
                        schemas=managed.schema_info,
                    )

                    # Update session state
                    st.session_state[KEY_CURRENT_DATASET] = managed
                    st.session_state[KEY_DATA_QUALITY_REPORT] = profile
                    st.session_state[KEY_DATASET_SCHEMA] = managed.schema_info
                    st.session_state[KEY_DATASET_METADATA] = DatasetMetadata(
                        filename=managed.filename,
                        file_format=managed.file_format,
                        row_count=len(managed.working_df),
                        column_count=len(managed.working_df.columns),
                        dataset_id=managed.dataset_id,
                        file_size_bytes=managed.file_size_bytes,
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

    # Display active dataset preview if loaded
    dataset: ManagedDataset | None = st.session_state.get(KEY_CURRENT_DATASET)
    if dataset is not None:
        st.markdown("---")
        st.markdown(f"### Active Working Dataset: `{dataset.filename}`")
        meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
        meta_c1.metric("Rows", f"{len(dataset.working_df):,}")
        meta_c2.metric("Columns", f"{len(dataset.working_df.columns):,}")
        meta_c3.metric("Size", f"{dataset.file_size_bytes / 1024:.1f} KB")
        meta_c4.metric("Fingerprint", dataset.fingerprint)

        st.caption("First 20 rows of active working dataset:")
        st.dataframe(dataset.working_df.head(20), use_container_width=True)
    else:
        st.info(
            "👆 Upload a CSV, Excel, or Parquet dataset above to activate the Quality Intelligence Engine."
        )

# ---------------------------------------------------------
# TAB 2: SCHEMA EXPLORER
# ---------------------------------------------------------
with tab_schema:
    st.subheader("Inferred Schema & Semantic Classification")
    st.caption("Auto-detected data types, cardinality, identifier roles, and target candidates.")

    dataset = st.session_state.get(KEY_CURRENT_DATASET)
    if dataset is None:
        st.info("No active dataset loaded. Please upload a file in the 'Upload & Preview' tab.")
    else:
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

        st.dataframe(schema_records, use_container_width=True)

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
# TAB 3: QUALITY OVERVIEW
# ---------------------------------------------------------
with tab_overview:
    st.subheader("Data Quality Score & Anomaly Summary")
    profile = st.session_state.get(KEY_DATA_QUALITY_REPORT)

    if profile is None:
        st.info("No active quality profile. Upload a dataset to inspect its health rating.")
    else:
        qs = profile.quality_score
        sc_col1, sc_col2, sc_col3, sc_col4, sc_col5 = st.columns(5)
        sc_col1.metric(
            "Overall Score", f"{qs.overall_score:.0f}/100", help="Weighted composite rating"
        )
        sc_col2.metric("Completeness (40%)", f"{qs.completeness_score:.1f}%")
        sc_col3.metric("Uniqueness (25%)", f"{qs.uniqueness_score:.1f}%")
        sc_col4.metric("Consistency (20%)", f"{qs.consistency_score:.1f}%")
        sc_col5.metric("Validity (15%)", f"{qs.validity_score:.1f}%")

        st.info(f"ℹ️ **Scoring Rationale:** {qs.explanation}")

        # Completeness Bar Chart
        st.markdown("#### Column Completeness Overview")
        col_completeness = [
            {
                "Column": col,
                "Completeness (%)": round(100.0 - s.null_percentage, 1),
                "Missing (%)": s.null_percentage,
            }
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

        # Quality Issues Table
        st.markdown("#### Detected Quality Issues")
        severity_filter = st.multiselect(
            "Filter issues by severity:",
            options=["critical", "high", "medium", "low"],
            default=["critical", "high", "medium"],
        )

        filtered_issues = [i for i in profile.quality_issues if i.severity in severity_filter]
        if filtered_issues:
            issue_rows = [
                {
                    "Severity": i.severity.upper(),
                    "Category": i.category.upper(),
                    "Title": i.title,
                    "Affected Columns": ", ".join(i.affected_columns),
                    "Affected Rows": f"{i.affected_row_count:,}",
                    "Evidence": i.measured_evidence,
                    "Recommendation": i.recommended_action,
                }
                for i in filtered_issues
            ]
            st.dataframe(issue_rows, use_container_width=True)
        else:
            st.success("No issues detected for the selected severity filters!")

        # Quality Certification Badge & Download
        st.markdown("---")
        st.markdown("#### 📜 Data Quality Certification & Audit Export")
        cert_payload = {
            "dataset_filename": dataset.filename,
            "fingerprint": dataset.fingerprint,
            "quality_grade": "A (Production Ready)"
            if qs.overall_score >= 90
            else ("B (Usable with Warnings)" if qs.overall_score >= 75 else "C (Needs Cleaning)"),
            "overall_score": round(qs.overall_score, 1),
            "completeness_score": round(qs.completeness_score, 1),
            "uniqueness_score": round(qs.uniqueness_score, 1),
            "consistency_score": round(qs.consistency_score, 1),
            "validity_score": round(qs.validity_score, 1),
            "total_rows": len(dataset.working_df),
            "total_columns": len(dataset.working_df.columns),
            "active_issues_count": len(profile.quality_issues),
            "applied_transformations": len(pipeline.records),
        }
        cert_json = json.dumps(cert_payload, indent=2)
        st.download_button(
            label="📥 Download Quality Audit Certificate (JSON)",
            data=cert_json,
            file_name=f"quality_audit_{dataset.dataset_id}.json",
            mime="application/json",
            key="btn_download_quality_cert",
        )

# ---------------------------------------------------------
# TAB 4: MISSINGNESS ANALYSIS
# ---------------------------------------------------------
with tab_missing:
    st.subheader("Missing-Value Deep Dive")
    dataset = st.session_state.get(KEY_CURRENT_DATASET)
    profile = st.session_state.get(KEY_DATA_QUALITY_REPORT)

    if dataset is None or profile is None:
        st.info("Upload a dataset to inspect missing value patterns.")
    else:
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

# ---------------------------------------------------------
# TAB 5: DUPLICATE ANALYSIS
# ---------------------------------------------------------
with tab_dups:
    st.subheader("Duplicate Record Analysis")
    dataset = st.session_state.get(KEY_CURRENT_DATASET)
    profile = st.session_state.get(KEY_DATA_QUALITY_REPORT)

    if dataset is None or profile is None:
        st.info("Upload a dataset to evaluate exact and key-based duplicates.")
    else:
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

# ---------------------------------------------------------
# TAB 6: OUTLIER ANALYSIS
# ---------------------------------------------------------
with tab_outliers:
    st.subheader("Outlier & Anomaly Diagnostics")
    dataset = st.session_state.get(KEY_CURRENT_DATASET)
    profile = st.session_state.get(KEY_DATA_QUALITY_REPORT)

    if dataset is None or profile is None:
        st.info("Upload a dataset to perform statistical outlier diagnostics.")
    else:
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
# TAB 7: CLEANING WORKBENCH
# ---------------------------------------------------------
with tab_clean:
    st.subheader("Data Cleaning Workbench")
    st.caption(
        "Preview and execute auditable, deterministic transformations. Original data is kept immutable."
    )

    dataset = st.session_state.get(KEY_CURRENT_DATASET)
    profile = st.session_state.get(KEY_DATA_QUALITY_REPORT)

    if dataset is None or profile is None:
        st.info("Upload a dataset to review cleaning recommendations.")
    else:
        # Comparison of Original vs Working dataset
        orig_rows = len(dataset._raw_df)
        curr_rows = len(dataset.working_df)
        orig_cols = len(dataset._raw_df.columns)
        curr_cols = len(dataset.working_df.columns)

        comp_c1, comp_c2, comp_c3, comp_c4 = st.columns(4)
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
        comp_c3.metric("Applied Transformations", f"{len(pipeline.records)}")
        comp_c4.metric("Current Quality Score", f"{profile.quality_score.overall_score:.0f}/100")

        st.markdown("---")
        st.markdown("#### 1. Recommended Cleaning Actions")

        actionable_issues = [
            i
            for i in profile.quality_issues
            if i.suggested_action_type is not None and not i.is_resolved
        ]

        if not actionable_issues:
            st.success("No automated cleaning transformations currently recommended!")
        else:
            # One-Click Auto-Clean for Safe Actions
            safe_issues = [
                i
                for i in actionable_issues
                if i.suggested_action_type in ("drop_duplicates", "strip_whitespace")
                or (i.suggested_action_type == "drop_column" and "100%" in i.title)
            ]
            if safe_issues:
                st.markdown("##### ⚡ One-Click Auto-Clean (Safe Non-Destructive Actions)")
                st.caption(
                    f"Found **{len(safe_issues)}** verified safe cleaning actions (whitespace trimming, row deduplication, dropping empty columns)."
                )
                if st.button(
                    "🚀 Apply All Safe Recommendations", key="btn_auto_clean_safe", type="primary"
                ):
                    with st.spinner("Applying safe cleaning transformations..."):
                        from src.profiling.models import TransformationSpec

                        applied_c = 0
                        for s_iss in safe_issues:
                            spec = TransformationSpec(
                                spec_id=f"auto_{s_iss.issue_id}",
                                action_type=s_iss.suggested_action_type,
                                column_name=s_iss.suggested_action_params.get("column_name"),
                                parameters=s_iss.suggested_action_params,
                                description=s_iss.recommended_action,
                            )
                            pipeline.apply(spec, dataset)
                            applied_c += 1

                        new_profile = profile_dataset(
                            dataset.working_df, dataset_id=dataset.dataset_id
                        )
                        st.session_state[KEY_DATA_QUALITY_REPORT] = new_profile
                        st.session_state[KEY_DATASET_SCHEMA] = dataset.schema_info
                        invalidate_downstream_analyses(st.session_state)
                        st.success(f"Successfully applied {applied_c} safe transformations!")
                        st.rerun()

                st.markdown("---")

            issue_options = {
                f"[{i.severity.upper()}] {i.title} → {i.recommended_action}": i
                for i in actionable_issues
            }
            selected_label = st.selectbox("Select issue to resolve:", list(issue_options.keys()))
            selected_issue = issue_options[selected_label]

            st.write(f"**Explanation:** {selected_issue.explanation}")
            st.write(f"**Consequences:** {selected_issue.potential_consequences}")

            from src.profiling.models import TransformationSpec

            spec_to_preview = TransformationSpec(
                spec_id=f"spec_{selected_issue.issue_id}",
                action_type=selected_issue.suggested_action_type,  # type: ignore[arg-type]
                column_name=selected_issue.suggested_action_params.get("column_name"),
                parameters=selected_issue.suggested_action_params,
                description=selected_issue.recommended_action,
            )

            # Dry-run Preview
            try:
                preview = preview_transformation(dataset.working_df, spec_to_preview)
                st.markdown("##### Transformation Dry-Run Preview")
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
                        # Re-profile working dataset
                        new_profile = profile_dataset(
                            dataset.working_df, dataset_id=dataset.dataset_id
                        )
                        st.session_state[KEY_DATA_QUALITY_REPORT] = new_profile
                        st.session_state[KEY_DATASET_SCHEMA] = dataset.schema_info
                        invalidate_downstream_analyses(st.session_state)
                        st.success(f"Applied: {record.summary}")
                        st.rerun()

            except Exception as e:
                st.error(f"Could not generate preview: {e}")

        # Transformation History & Controls
        st.markdown("---")
        st.markdown("#### 2. Transformation History & Audit Trail")

        if pipeline.records:
            history_rows = [
                {
                    "Index": idx + 1,
                    "Action": r.spec.action_type,
                    "Target": r.spec.column_name or "Dataset",
                    "Rows (Before → After)": f"{r.rows_before:,} → {r.rows_after:,}",
                    "Cells Modified": f"{r.cells_modified:,}",
                    "Summary": r.summary,
                    "Timestamp": r.applied_at.strftime("%H:%M:%S UTC"),
                }
                for idx, r in enumerate(pipeline.records)
            ]
            st.dataframe(history_rows, use_container_width=True)

            ctrl_c1, ctrl_c2 = st.columns(2)
            with ctrl_c1:
                if st.button("↩️ Undo Last Transformation", key="btn_undo"):
                    undone = pipeline.undo(dataset)
                    if undone:
                        new_profile = profile_dataset(
                            dataset.working_df, dataset_id=dataset.dataset_id
                        )
                        st.session_state[KEY_DATA_QUALITY_REPORT] = new_profile
                        st.session_state[KEY_DATASET_SCHEMA] = dataset.schema_info
                        invalidate_downstream_analyses(st.session_state)
                        st.info(f"Reverted: {undone.summary}")
                        st.rerun()
            with ctrl_c2:
                if st.button("🔄 Reset All to Original Raw Dataset", key="btn_reset_all"):
                    pipeline.reset(dataset)
                    new_profile = profile_dataset(dataset.working_df, dataset_id=dataset.dataset_id)
                    st.session_state[KEY_DATA_QUALITY_REPORT] = new_profile
                    st.session_state[KEY_DATASET_SCHEMA] = dataset.schema_info
                    invalidate_downstream_analyses(st.session_state)
                    st.warning(
                        "All transformations reverted! Working dataset restored to immutable raw state."
                    )
                    st.rerun()
        else:
            st.caption(
                "No transformations applied yet. Dataset is currently in its original raw state."
            )

st.markdown("---")
st.page_link("app.py", label="← Return to Main Workspace", icon="🏠")
