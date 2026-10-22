"""Page 2: Data Quality Lab - Ingestion, profiling, issue audit, and interactive cleaning workbench."""

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
            st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(m_fig, use_container_width=True)

        st.dataframe(missing_data, use_container_width=True)

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

        # Key-based duplicate checker
        st.markdown("---")
        st.markdown("#### Entity Key Uniqueness Checker")
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
            fig.update_layout(margin={"l": 20, "r": 20, "t": 40, "b": 20}, height=350)
            st.plotly_chart(fig, use_container_width=True)

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
