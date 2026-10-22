"""Deterministic data cleaning transformations with preview, execution, and audit logging."""

import uuid

import pandas as pd

from src.core.exceptions import AnalyticalExecutionError
from src.profiling.models import (
    TransformationPreview,
    TransformationRecord,
    TransformationSpec,
    TransformationType,
)


def preview_transformation(df: pd.DataFrame, spec: TransformationSpec) -> TransformationPreview:
    """Simulate a proposed transformation on a cloned copy and generate before/after preview."""
    rows_before = len(df)
    cols_before = len(df.columns)
    warnings: list[str] = []

    if (
        spec.column_name
        and spec.column_name not in df.columns
        and spec.action_type != "drop_duplicates"
    ):
        raise AnalyticalExecutionError(
            message=f"Column '{spec.column_name}' does not exist in dataset.",
            user_message=f"Target column '{spec.column_name}' was not found in the current dataset.",
        )

    # Dry run on a shallow/deep copy
    working_copy = df.copy(deep=True)
    transformed_df, record = apply_transformation(working_copy, spec)

    rows_after = len(transformed_df)
    cols_after = len(transformed_df.columns)

    # Extract sample preview (up to 4 rows from affected records)
    before_sample: list[dict[str, object]] = []
    after_sample: list[dict[str, object]] = []

    if (
        spec.column_name
        and spec.column_name in df.columns
        and spec.column_name in transformed_df.columns
    ):
        diff_mask = df[spec.column_name].ne(transformed_df[spec.column_name]) | (
            df[spec.column_name].isna() != transformed_df[spec.column_name].isna()
        )
        changed_indices = df[diff_mask].index[:4]
        if len(changed_indices) > 0:
            before_sample = df.loc[changed_indices, [spec.column_name]].to_dict(orient="records")
            after_sample = transformed_df.loc[changed_indices, [spec.column_name]].to_dict(
                orient="records"
            )
    elif spec.action_type in ("drop_duplicates", "drop_missing_rows"):
        dropped_count = rows_before - rows_after
        if dropped_count > (rows_before * 0.3):
            warnings.append(f"Operation will remove {dropped_count} rows (>{30}% of dataset).")

    return TransformationPreview(
        action_type=spec.action_type,
        column_name=spec.column_name,
        rows_before=rows_before,
        rows_after=rows_after,
        cols_before=cols_before,
        cols_after=cols_after,
        affected_rows_count=record.rows_before - record.rows_after
        if record.rows_before != record.rows_after
        else record.cells_modified,
        affected_cells_count=record.cells_modified,
        before_sample=before_sample,
        after_sample=after_sample,
        warnings=warnings,
        summary=record.summary,
    )


def apply_transformation(
    df: pd.DataFrame,
    spec: TransformationSpec,
) -> tuple[pd.DataFrame, TransformationRecord]:
    """Execute an approved deterministic transformation and return transformed DataFrame and audit record."""
    target_df = df.copy(deep=True)
    rows_before = len(target_df)
    cols_before = len(target_df.columns)
    cells_modified = 0
    summary = ""

    action: TransformationType = spec.action_type
    col = spec.column_name
    params = spec.parameters or {}

    if action == "drop_column":
        if not col or col not in target_df.columns:
            raise AnalyticalExecutionError(f"Cannot drop missing column: '{col}'")
        target_df.drop(columns=[col], inplace=True)
        summary = f"Dropped column '{col}'."

    elif action == "drop_duplicates":
        subset = params.get("subset")
        keep = params.get("keep", "first")
        dups_before = target_df.duplicated(subset=subset, keep=keep).sum()
        target_df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        summary = f"Removed {dups_before} duplicate rows (keep='{keep}')."

    elif action == "drop_missing_rows":
        subset = [col] if col else params.get("subset")
        initial_len = len(target_df)
        target_df.dropna(subset=subset, inplace=True)
        dropped = initial_len - len(target_df)
        summary = f"Dropped {dropped} rows with missing values in {subset or 'any column'}."

    elif action == "impute_numeric":
        if not col or col not in target_df.columns:
            raise AnalyticalExecutionError(f"Target column '{col}' missing for imputation.")
        strategy = params.get("strategy", "median")
        null_mask = target_df[col].isna()
        cells_modified = int(null_mask.sum())
        clean_num = target_df[col].dropna().astype(float)

        if cells_modified > 0 and len(clean_num) > 0:
            if strategy == "median":
                fill_val = float(clean_num.median())
            elif strategy == "mean":
                fill_val = float(clean_num.mean())
            elif strategy == "constant":
                fill_val = float(params.get("fill_value", 0.0))
            else:
                fill_val = 0.0

            target_df[col] = target_df[col].fillna(fill_val)
            summary = f"Imputed {cells_modified} nulls in '{col}' with {strategy} ({fill_val:.2f})."
        else:
            summary = f"No null values required imputation in '{col}'."

    elif action == "impute_categorical":
        if not col or col not in target_df.columns:
            raise AnalyticalExecutionError(
                f"Target column '{col}' missing for categorical imputation."
            )
        strategy = params.get("strategy", "mode")
        null_mask = target_df[col].isna()
        cells_modified = int(null_mask.sum())

        if cells_modified > 0:
            if strategy == "mode":
                mode_series = target_df[col].mode()
                fill_val = str(mode_series.iloc[0]) if not mode_series.empty else "Missing"
            else:
                fill_val = str(params.get("fill_value", "Missing"))

            target_df[col] = target_df[col].fillna(fill_val)
            summary = f"Imputed {cells_modified} nulls in '{col}' with {strategy} ('{fill_val}')."
        else:
            summary = f"No null values in '{col}'."

    elif action == "strip_whitespace":
        if not col or col not in target_df.columns:
            raise AnalyticalExecutionError(f"Target column '{col}' missing.")
        series = target_df[col].astype(str)
        stripped = series.str.strip()
        cells_modified = int((series != stripped).sum())
        target_df[col] = stripped
        summary = f"Stripped leading/trailing whitespace across {cells_modified} values in '{col}'."

    elif action == "standardize_casing":
        if not col or col not in target_df.columns:
            raise AnalyticalExecutionError(f"Target column '{col}' missing.")
        case_type = params.get("case", "title")
        series = target_df[col].dropna().astype(str)

        if case_type == "lower":
            standardized = series.str.lower()
        elif case_type == "upper":
            standardized = series.str.upper()
        else:
            standardized = series.str.title()

        cells_modified = int((series != standardized).sum())
        target_df.loc[series.index, col] = standardized
        summary = f"Standardized casing to {case_type} across {cells_modified} records in '{col}'."

    elif action == "cap_outliers":
        if not col or col not in target_df.columns:
            raise AnalyticalExecutionError(f"Target column '{col}' missing.")
        lower_q = float(params.get("lower_percentile", 0.01))
        upper_q = float(params.get("upper_percentile", 0.99))
        clean_num = target_df[col].dropna().astype(float)

        lower_bound = float(clean_num.quantile(lower_q))
        upper_bound = float(clean_num.quantile(upper_q))

        capped_series = clean_num.clip(lower=lower_bound, upper=upper_bound)
        cells_modified = int((clean_num != capped_series).sum())
        target_df.loc[clean_num.index, col] = capped_series
        summary = (
            f"Winsorized/capped {cells_modified} outlier values in '{col}' "
            f"to quantiles [{lower_q:.2f}, {upper_q:.2f}]."
        )

    elif action == "convert_type":
        if not col or col not in target_df.columns:
            raise AnalyticalExecutionError(f"Target column '{col}' missing.")
        target_type = params.get("target_type", "datetime")
        initial_nulls = target_df[col].isna().sum()

        if target_type == "datetime":
            target_df[col] = pd.to_datetime(target_df[col], errors="coerce")
            new_nulls = target_df[col].isna().sum() - initial_nulls
            summary = (
                f"Converted '{col}' to datetime64 ({new_nulls} unparseable values coerced to null)."
            )
        elif target_type == "numeric":
            target_df[col] = pd.to_numeric(target_df[col], errors="coerce")
            new_nulls = target_df[col].isna().sum() - initial_nulls
            summary = (
                f"Converted '{col}' to numeric ({new_nulls} unparseable values coerced to null)."
            )
        elif target_type == "boolean":
            target_df[col] = target_df[col].astype(bool)
            summary = f"Converted '{col}' to boolean."

    else:
        raise AnalyticalExecutionError(f"Unsupported transformation action: '{action}'")

    record = TransformationRecord(
        record_id=f"rec_{uuid.uuid4().hex[:8]}",
        spec=spec,
        rows_before=rows_before,
        rows_after=len(target_df),
        cols_before=cols_before,
        cols_after=len(target_df.columns),
        cells_modified=cells_modified,
        summary=summary,
    )

    return target_df, record
