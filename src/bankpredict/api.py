
import json
import os

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from fastapi import (
    FastAPI,
    Header,
    HTTPException,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from bankpredict.schema import FEATURE_COLUMNS


APP_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = Path(
    os.getenv(
        "BANKPREDICT_MODEL_PATH",
        str(
            APP_ROOT
            / "artifacts"
            / "model.joblib"
        ),
    )
)

METADATA_FILE = Path(
    os.getenv(
        "BANKPREDICT_METADATA_PATH",
        str(
            APP_ROOT
            / "artifacts"
            / "model_metadata.json"
        ),
    )
)

app = FastAPI(
    title="BankPredict API",
    version="1.0.0",
    description=(
        "Bank marketing subscription prediction API"
    ),
)


class CustomerRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    age: int = Field(ge=18, le=110)
    job: str = Field(min_length=1)
    marital: str = Field(min_length=1)
    education: str = Field(min_length=1)
    default: str = Field(min_length=1)
    balance: float
    housing: str = Field(min_length=1)
    loan: str = Field(min_length=1)
    contact: str = Field(min_length=1)
    day: int = Field(ge=1, le=31)
    month: str = Field(min_length=3)
    campaign: int = Field(ge=1)
    pdays: int = Field(ge=-1)
    previous: int = Field(ge=0)
    poutcome: str = Field(min_length=1)


@lru_cache(maxsize=1)
def load_bundle():
    if (
        not MODEL_FILE.is_file()
        or not METADATA_FILE.is_file()
    ):
        raise FileNotFoundError(
            "Train the model before starting the API"
        )

    model = joblib.load(
        MODEL_FILE
    )

    metadata = json.loads(
        METADATA_FILE.read_text(
            encoding="utf-8"
        )
    )

    return model, metadata


@app.get("/")
def root():
    return {
        "service": "BankPredict",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    try:
        _, metadata = load_bundle()

    except (
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return {
        "status": "ok",
        "model_name": metadata["model_name"],
    }


@app.get("/model-info")
def model_info():
    _, metadata = load_bundle()

    return {
        "model_name": metadata["model_name"],
        "version": metadata["model_version"],
        "number_of_features": len(
            metadata["features"]
        ),
    }


@app.post("/predict")
def predict(
    customer: CustomerRequest,
    x_api_key: str | None = Header(default=None),
):
    required_key = os.getenv(
        "API_KEY",
        "",
    )

    if required_key and x_api_key != required_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    model, metadata = load_bundle()

    values = customer.model_dump()

    data = pd.DataFrame([
        {
            name: values[name]
            for name in FEATURE_COLUMNS
        }
    ])

    probability = float(
        model.predict_proba(data)[0, 1]
    )

    threshold = float(
        metadata["decision_threshold"]
    )

    return {
        "subscription_probability": round(
            probability,
            5,
        ),
        "predicted_subscription": (
            probability >= threshold
        ),
        "decision_threshold": round(
            threshold,
            5,
        ),
        "recommend_contact": (
            probability >= threshold
        ),
        "model_name": metadata["model_name"],
    }
