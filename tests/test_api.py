
import json

from fastapi.testclient import TestClient
import pytest

from bankpredict.api import (
    APP_ROOT,
    app,
    load_bundle,
)


@pytest.fixture()
def example_customer():
    return json.loads(
        (
            APP_ROOT
            / "examples"
            / "predict.json"
        ).read_text(
            encoding="utf-8"
        )
    )


def test_missing_required_field_returns_422(
    example_customer,
):
    payload = dict(example_customer)

    payload.pop("age")

    response = TestClient(app).post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_predict_when_model_is_trained(
    example_customer,
    monkeypatch,
):
    model_path = (
        APP_ROOT
        / "artifacts"
        / "model.joblib"
    )

    if not model_path.exists():
        pytest.skip(
            "Train the model before running this test"
        )

    monkeypatch.delenv(
        "API_KEY",
        raising=False,
    )

    load_bundle.cache_clear()

    client = TestClient(app)

    response = client.post(
        "/predict",
        json=example_customer,
    )

    assert response.status_code == 200

    assert (
        0
        <= response.json()[
            "subscription_probability"
        ]
        <= 1
    )

    health_response = client.get(
        "/health"
    )

    assert health_response.status_code == 200
