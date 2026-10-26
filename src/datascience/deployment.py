"""Deployment helpers: a FastAPI service template for an exported model."""

from __future__ import annotations

from src.datascience.modeling import FittedModel


def _py_type(info: dict) -> str:
    return "float" if info.get("kind") == "numeric" else "str"


def fastapi_template(model: FittedModel, model_path: str = "model.joblib") -> str:
    """A small FastAPI app that loads the exported model and serves /predict and /health."""
    fields = [
        f"    {feat!r}: ({_py_type(model.feature_info.get(feat, {}))}, {model.feature_info.get(feat, {}).get('default')!r}),"
        for feat in model.features
    ]
    model_def = 'Record = create_model("Record", **{\n' + "\n".join(fields) + "\n})"
    classes = [c.item() if hasattr(c, "item") else c for c in (model.classes or [])]
    proba = (
        f"""
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(frame)
        out["probabilities"] = [dict(zip({classes!r}, map(float, row))) for row in proba]"""
        if classes
        else ""
    )
    return f'''"""Prediction service for {model.name}. Run:  uvicorn app:app --reload"""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import create_model

bundle = joblib.load({model_path!r})   # only load model files you trust (joblib uses pickle)
pipeline, meta = bundle["pipeline"], bundle["meta"]

{model_def}

app = FastAPI(title={model.name!r})


@app.get("/health")
def health():
    return {{"status": "ok", "model": meta["name"], "target": meta["target"], "metrics": meta["metrics"]}}


@app.post("/predict")
def predict(records: list[Record]):
    frame = pd.DataFrame([r.model_dump() for r in records])[meta["features"]]
    out = {{"predictions": pipeline.predict(frame).tolist()}}{proba}
    return out
'''


def requirements_text(model: FittedModel) -> str:
    extra = []
    name = model.estimator_name
    if "XGBoost" in name:
        extra.append("xgboost")
    if "LightGBM" in name:
        extra.append("lightgbm")
    if "CatBoost" in name:
        extra.append("catboost")
    return (
        "\n".join(["fastapi", "uvicorn", "pandas", "numpy", "scikit-learn", "joblib", *extra])
        + "\n"
    )
