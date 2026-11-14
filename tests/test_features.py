
import pandas as pd

from bankpredict.features import add_contact_features
from bankpredict.model import build_model
from bankpredict.schema import FEATURE_COLUMNS


def test_pdays_sentinel_is_encoded_correctly():
    sample = pd.DataFrame({
        "pdays": [-1, 20, 0]
    })

    features = add_contact_features(sample)

    assert (
        features["previously_contacted"].tolist()
        == [0, 1, 1]
    )

    assert pd.isna(
        features.loc[
            0,
            "days_since_previous_contact",
        ]
    )

    assert (
        features.loc[
            1,
            "days_since_previous_contact",
        ]
        == 20
    )


def test_model_pipeline_can_fit_and_predict():
    first = {
        "age": 30,
        "job": "technician",
        "marital": "single",
        "education": "tertiary",
        "default": "no",
        "balance": 500,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 10,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }

    second = {
        **first,
        "job": "management",
        "balance": 2500,
        "housing": "no",
        "pdays": 25,
        "previous": 2,
    }

    x = pd.DataFrame(
        [first, second, first, second],
        columns=FEATURE_COLUMNS,
    )

    y = [0, 1, 0, 1]

    model = build_model(
        "logistic_regression"
    )

    model.fit(x, y)

    probabilities = model.predict_proba(
        x
    )[:, 1]

    assert len(probabilities) == 4

    assert all(
        0 <= value <= 1
        for value in probabilities
    )
