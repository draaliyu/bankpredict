
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from xgboost import XGBClassifier

from bankpredict.features import add_contact_features
from bankpredict.schema import CATEGORICAL_COLUMNS


MODEL_NUMERIC = [
    "age",
    "balance",
    "day",
    "campaign",
    "previous",
    "previously_contacted",
    "days_since_previous_contact",
]


def build_model(name: str):
    numeric = Pipeline([
        (
            "fill_numeric",
            SimpleImputer(strategy="median"),
        ),
        (
            "scale_numeric",
            StandardScaler(),
        ),
    ])

    categorical = Pipeline([
        (
            "one_hot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ])

    preprocessing = ColumnTransformer([
        (
            "numeric",
            numeric,
            MODEL_NUMERIC,
        ),
        (
            "categorical",
            categorical,
            CATEGORICAL_COLUMNS,
        ),
    ])

    if name == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=2000,
            random_state=42,
        )

    elif name == "xgboost":
        estimator = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            random_state=42,
        )

    else:
        raise ValueError(
            f"Unknown model: {name}"
        )

    return Pipeline([
        (
            "engineer",
            FunctionTransformer(
                add_contact_features,
                validate=False,
            ),
        ),
        (
            "preprocess",
            preprocessing,
        ),
        (
            "classifier",
            estimator,
        ),
    ])
