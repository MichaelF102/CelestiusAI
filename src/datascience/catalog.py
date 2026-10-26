"""Model catalog: 20+ algorithms for regression and classification with model cards.

Each entry knows how to build the estimator, which hyperparameters are worth tuning, and when
the algorithm is a good (or poor) choice. Optional boosters (XGBoost, LightGBM, CatBoost) are
listed only when their package is installed.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

REGRESSION = "regression"
CLASSIFICATION = "classification"


def task_group(task: str) -> str:
    """Map 'binary_classification' / 'multiclass_classification' / 'regression' to a group."""
    return REGRESSION if task == REGRESSION else CLASSIFICATION


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@dataclass(frozen=True)
class ModelCard:
    when_to_use: str
    strengths: str
    weaknesses: str
    interpretability: int  # 1 (black box) .. 5 (fully transparent)
    speed: int  # 1 (slow) .. 5 (fast)
    needs_scaling: bool = False


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    tasks: tuple[str, ...]
    factory: Callable[[str, int], Any]  # (task group, seed) -> estimator
    space: dict[str, list[Any]]
    card: ModelCard
    package: str | None = None  # optional dependency
    note: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def available(self) -> bool:
        return self.package is None or has_package(self.package)


class EncodedClassifier(ClassifierMixin, BaseEstimator):
    """Wraps a classifier that needs integer labels 0..k-1 (e.g. XGBoost) so any labels work."""

    def __init__(self, estimator: Any = None) -> None:
        self.estimator = estimator

    def fit(self, X: Any, y: Any) -> EncodedClassifier:
        self.encoder_ = LabelEncoder().fit(y)
        self.estimator_ = clone(self.estimator).fit(X, self.encoder_.transform(y))
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, X: Any) -> Any:
        return self.encoder_.inverse_transform(self.estimator_.predict(X).astype(int))

    def predict_proba(self, X: Any) -> Any:
        return self.estimator_.predict_proba(X)


def _sk_version() -> tuple[int, int]:
    major, minor = sklearn.__version__.split(".")[:2]
    return int(major), int(minor)


def _l1_logistic(seed: int) -> Any:
    """L1-penalized logistic regression across scikit-learn's penalty -> l1_ratio change (1.8)."""
    if _sk_version() >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, solver="saga", C=1.0, max_iter=5000)
    return LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000)


def _svc(seed: int) -> Any:
    """SVC with probabilities; scikit-learn 1.9 deprecates probability=True."""
    if _sk_version() >= (1, 9):
        return CalibratedClassifierCV(SVC(C=1.0, random_state=seed), ensemble=False, cv=3)
    return SVC(C=1.0, probability=True, random_state=seed)


def _mk(
    reg: Callable[..., Any] | None, clf: Callable[..., Any] | None
) -> Callable[[str, int], Any]:
    def factory(group: str, seed: int) -> Any:
        maker = reg if group == REGRESSION else clf
        if maker is None:
            raise ValueError("This model does not support the requested task.")
        return maker(seed)

    return factory


_BOTH = (REGRESSION, CLASSIFICATION)
_TREE_SPACE = {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 5, 10, 20]}
_FOREST_SPACE = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", 0.5, 1.0],
}


def _xgb(reg: bool) -> Callable[[int], Any]:
    def make(seed: int) -> Any:
        import xgboost as xgb

        if reg:
            return xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
            )
        return EncodedClassifier(
            xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
                eval_metric="logloss",
            )
        )

    return make


def _lgbm(reg: bool) -> Callable[[int], Any]:
    def make(seed: int) -> Any:
        import lightgbm as lgb

        cls = lgb.LGBMRegressor if reg else lgb.LGBMClassifier
        return cls(
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    return make


def _catboost(reg: bool) -> Callable[[int], Any]:
    def make(seed: int) -> Any:
        import catboost as cb

        cls = cb.CatBoostRegressor if reg else cb.CatBoostClassifier
        return cls(iterations=300, random_seed=seed, verbose=0)

    return make


CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        "Baseline",
        "baseline",
        _BOTH,
        _mk(
            lambda s: DummyRegressor(strategy="mean"),
            lambda s: DummyClassifier(strategy="most_frequent"),
        ),
        {},
        ModelCard(
            "Always run it: any real model must beat it.",
            "Instant; sets the bar.",
            "Learns nothing.",
            5,
            5,
        ),
        aliases=("Baseline (mean)", "Baseline (most frequent)"),
    ),
    ModelSpec(
        "Linear / Logistic Regression",
        "linear",
        _BOTH,
        _mk(lambda s: LinearRegression(), lambda s: LogisticRegression(max_iter=2000)),
        {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ModelCard(
            "A transparent first model; coefficients you can explain.",
            "Fast, stable, interpretable.",
            "Misses non-linear patterns and interactions.",
            5,
            5,
            True,
        ),
        aliases=("Linear Regression", "Logistic Regression"),
    ),
    ModelSpec(
        "Ridge / L2 Logistic",
        "linear",
        _BOTH,
        _mk(lambda s: Ridge(alpha=1.0), lambda s: LogisticRegression(C=1.0, max_iter=2000)),
        {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0], "C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ModelCard(
            "Many correlated features or few rows.",
            "Stable under collinearity.",
            "Keeps every feature.",
            4,
            5,
            True,
        ),
        aliases=("Ridge",),
    ),
    ModelSpec(
        "Lasso / L1 Logistic",
        "linear",
        _BOTH,
        _mk(
            lambda s: Lasso(alpha=0.01, max_iter=20000),
            lambda s: LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000),
        ),
        {"alpha": [0.001, 0.01, 0.1, 1.0], "C": [0.01, 0.1, 1.0, 10.0]},
        ModelCard(
            "You suspect only a few features matter and want them selected.",
            "Built-in feature selection.",
            "Unstable when features are strongly correlated.",
            5,
            4,
            True,
        ),
        aliases=("Lasso",),
    ),
    ModelSpec(
        "ElasticNet",
        "linear",
        (REGRESSION,),
        _mk(lambda s: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000), None),
        {"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
        ModelCard(
            "Correlated features and you still want some selection.",
            "Blends Lasso and Ridge.",
            "Two hyperparameters to tune.",
            4,
            4,
            True,
        ),
    ),
    ModelSpec(
        "Huber (robust)",
        "linear",
        (REGRESSION,),
        _mk(lambda s: HuberRegressor(max_iter=500), None),
        {"epsilon": [1.1, 1.35, 1.75, 2.5], "alpha": [0.0001, 0.001, 0.01]},
        ModelCard(
            "Outliers in the target would distort ordinary least squares.",
            "Down-weights outliers.",
            "Slower than OLS; still linear.",
            5,
            4,
            True,
        ),
    ),
    ModelSpec(
        "Polynomial (degree 2) + Ridge",
        "linear",
        (REGRESSION,),
        _mk(
            lambda s: make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=1.0)),
            None,
        ),
        {"ridge__alpha": [0.1, 1.0, 10.0, 100.0], "polynomialfeatures__degree": [2, 3]},
        ModelCard(
            "Gentle curves and interactions between a few numeric features.",
            "Captures curvature while staying smooth.",
            "Feature count explodes with many columns.",
            3,
            3,
            True,
        ),
    ),
    ModelSpec(
        "Naive Bayes",
        "bayes",
        (CLASSIFICATION,),
        _mk(None, lambda s: GaussianNB()),
        {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
        ModelCard(
            "Tiny data sets and a fast probabilistic baseline.",
            "Very fast; works with few rows.",
            "Assumes features are independent.",
            4,
            5,
        ),
    ),
    ModelSpec(
        "Linear Discriminant Analysis",
        "linear",
        (CLASSIFICATION,),
        _mk(None, lambda s: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        {"shrinkage": [None, "auto", 0.1, 0.5]},
        ModelCard(
            "Roughly Gaussian classes with similar spread.",
            "Stable and fast; gives class projections.",
            "Struggles with strongly non-Gaussian data.",
            4,
            5,
            True,
        ),
    ),
    ModelSpec(
        "K-Nearest Neighbors",
        "neighbors",
        _BOTH,
        _mk(
            lambda s: KNeighborsRegressor(n_neighbors=7),
            lambda s: KNeighborsClassifier(n_neighbors=7),
        ),
        {"n_neighbors": [3, 5, 9, 15, 25], "weights": ["uniform", "distance"]},
        ModelCard(
            "Small, low-dimensional data where similar rows behave alike.",
            "No training; flexible boundaries.",
            "Slow on big data; hurt by irrelevant features.",
            3,
            2,
            True,
        ),
        aliases=("KNN",),
    ),
    ModelSpec(
        "Support Vector Machine (RBF)",
        "svm",
        _BOTH,
        _mk(lambda s: SVR(C=10.0), _svc),
        {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto"],
            "estimator__C": [0.1, 1.0, 10.0, 100.0],
            "estimator__gamma": ["scale", "auto"],
        },
        ModelCard(
            "Medium-sized data with smooth non-linear patterns.",
            "Strong accuracy on moderate data.",
            "Slow beyond ~20k rows; needs scaling.",
            2,
            2,
            True,
        ),
        aliases=("SVM",),
    ),
    ModelSpec(
        "Decision Tree",
        "tree",
        _BOTH,
        _mk(
            lambda s: DecisionTreeRegressor(max_depth=6, random_state=s),
            lambda s: DecisionTreeClassifier(max_depth=6, random_state=s),
        ),
        _TREE_SPACE,
        ModelCard(
            "You need rules a person can read.",
            "Readable if/else rules; no scaling.",
            "Unstable and overfits easily.",
            5,
            5,
        ),
    ),
    ModelSpec(
        "Random Forest",
        "ensemble",
        _BOTH,
        _mk(
            lambda s: RandomForestRegressor(n_estimators=150, random_state=s, n_jobs=-1),
            lambda s: RandomForestClassifier(n_estimators=150, random_state=s, n_jobs=-1),
        ),
        _FOREST_SPACE,
        ModelCard(
            "A strong default for tabular data with mixed features.",
            "Robust, little tuning, handles interactions.",
            "Larger and slower than linear models.",
            3,
            3,
        ),
    ),
    ModelSpec(
        "Extra Trees",
        "ensemble",
        _BOTH,
        _mk(
            lambda s: ExtraTreesRegressor(n_estimators=200, random_state=s, n_jobs=-1),
            lambda s: ExtraTreesClassifier(n_estimators=200, random_state=s, n_jobs=-1),
        ),
        _FOREST_SPACE,
        ModelCard(
            "Noisy data where random forests overfit.",
            "Faster than a forest; smoother fits.",
            "Slightly less accurate on clean signals.",
            3,
            3,
        ),
    ),
    ModelSpec(
        "Gradient Boosting",
        "boosting",
        _BOTH,
        _mk(
            lambda s: GradientBoostingRegressor(random_state=s),
            lambda s: GradientBoostingClassifier(random_state=s),
        ),
        {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.7, 1.0],
        },
        ModelCard(
            "Small to medium tabular data where accuracy matters most.",
            "High accuracy; flexible loss functions.",
            "Slower than histogram boosting; sensitive to settings.",
            2,
            2,
        ),
    ),
    ModelSpec(
        "Hist Gradient Boosting",
        "boosting",
        _BOTH,
        _mk(
            lambda s: HistGradientBoostingRegressor(random_state=s),
            lambda s: HistGradientBoostingClassifier(random_state=s),
        ),
        {
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "max_depth": [None, 3, 6, 10],
            "max_iter": [100, 200, 300],
            "l2_regularization": [0.0, 0.1, 1.0],
        },
        ModelCard(
            "Larger data sets; fast and accurate with missing values.",
            "Fast on big data; native missing support.",
            "Many knobs; less interpretable.",
            2,
            4,
        ),
        aliases=("Hist Gradient Boosting",),
    ),
    ModelSpec(
        "AdaBoost",
        "boosting",
        _BOTH,
        _mk(
            lambda s: AdaBoostRegressor(random_state=s),
            lambda s: AdaBoostClassifier(random_state=s),
        ),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.1, 0.5, 1.0]},
        ModelCard(
            "Weak learners combined into a stronger one; clean data.",
            "Simple boosting idea; few settings.",
            "Sensitive to noise and outliers.",
            3,
            3,
        ),
    ),
    ModelSpec(
        "Neural Network (MLP)",
        "neural",
        _BOTH,
        _mk(
            lambda s: MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=400, early_stopping=True, random_state=s
            ),
            lambda s: MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=400, early_stopping=True, random_state=s
            ),
        ),
        {
            "hidden_layer_sizes": [(32,), (64,), (64, 32)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.003, 0.01],
        },
        ModelCard(
            "Large data with complex patterns; try after simpler models.",
            "Flexible function approximator.",
            "Needs scaling and tuning; hard to explain.",
            1,
            2,
            True,
        ),
        aliases=("MLP",),
    ),
    ModelSpec(
        "XGBoost",
        "boosting",
        _BOTH,
        _mk(_xgb(True), _xgb(False)),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
        },
        ModelCard(
            "Competition-grade accuracy on tabular data.",
            "Very strong defaults and regularization.",
            "Extra dependency; less interpretable.",
            2,
            3,
        ),
        package="xgboost",
    ),
    ModelSpec(
        "LightGBM",
        "boosting",
        _BOTH,
        _mk(_lgbm(True), _lgbm(False)),
        {
            "n_estimators": [100, 200, 400],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.03, 0.05, 0.1],
            "min_child_samples": [10, 20, 40],
        },
        ModelCard(
            "Large data sets; very fast training.",
            "Fast, memory-efficient, accurate.",
            "Can overfit small data; extra dependency.",
            2,
            4,
        ),
        package="lightgbm",
    ),
    ModelSpec(
        "CatBoost",
        "boosting",
        _BOTH,
        _mk(_catboost(True), _catboost(False)),
        {"depth": [4, 6, 8], "learning_rate": [0.03, 0.1], "iterations": [200, 400]},
        ModelCard(
            "Data with many categorical features.",
            "Strong defaults; good with categories.",
            "Slower to train; extra dependency.",
            2,
            2,
        ),
        package="catboost",
    ),
)

BY_NAME: dict[str, ModelSpec] = {}
for _spec in CATALOG:
    BY_NAME[_spec.name] = _spec
    for _alias in _spec.aliases:
        BY_NAME.setdefault(_alias, _spec)


def get(name: str) -> ModelSpec:
    try:
        return BY_NAME[name]
    except KeyError as err:
        raise KeyError(f"Unknown model '{name}'.") from err


def available_models(
    task: str, include_optional: bool = True, include_baseline: bool = True
) -> list[ModelSpec]:
    group = task_group(task)
    out = []
    for spec in CATALOG:
        if group not in spec.tasks or not spec.available:
            continue
        if spec.package and not include_optional:
            continue
        if spec.family == "baseline" and not include_baseline:
            continue
        out.append(spec)
    return out


def display_name(spec: ModelSpec, task: str) -> str:
    """Task-appropriate label: 'Linear / Logistic Regression' -> 'Linear Regression'."""
    if " / " in spec.name:
        reg, clf = spec.name.split(" / ", 1)
        if reg == "Linear":
            return "Linear Regression" if task_group(task) == REGRESSION else "Logistic Regression"
        return reg if task_group(task) == REGRESSION else clf
    return spec.name


def build(name: str, task: str, seed: int = 42, params: dict[str, Any] | None = None) -> Any:
    spec = get(name)
    group = task_group(task)
    if group not in spec.tasks:
        raise ValueError(f"{name} does not support {group}.")
    if not spec.available:
        raise ImportError(
            f"{name} needs the '{spec.package}' package (pip install {spec.package})."
        )
    est = spec.factory(group, seed)
    if params:
        est.set_params(**params)
    return est


def param_space(name: str, task: str) -> dict[str, list[Any]]:
    """Tunable hyperparameters that actually exist on the estimator for this task."""
    spec = get(name)
    est = spec.factory(task_group(task), 0) if spec.available else None
    if est is None:
        return {}
    valid = set(est.get_params())
    return {k: list(v) for k, v in spec.space.items() if k in valid}


def tunable(task: str) -> list[str]:
    return [s.name for s in available_models(task) if param_space(s.name, task)]


def complexity_rank(name: str) -> float:
    """Lower = simpler/more interpretable (used by the one-standard-error rule)."""
    if name.startswith(("Stacking", "Voting", "Bagging")):
        return 7.5  # ensembles are the hardest to explain
    try:
        card = get(name).card
    except KeyError:
        return 6.0
    return (6 - card.interpretability) + (6 - card.speed) * 0.25


def optional_status() -> dict[str, bool]:
    return {s.name: s.available for s in CATALOG if s.package}
