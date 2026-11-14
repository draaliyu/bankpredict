
import argparse

import pandas as pd
from sqlalchemy import text

from bankpredict.config import (
    PROCESSED_CSV,
    database_engine,
)

from bankpredict.schema import (
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
)


SELECT_SQL = """
SELECT
    record_id,
    age,
    job,
    marital,
    education,
    default_status AS "default",
    balance,
    housing,
    loan,
    contact,
    day,
    month,
    campaign,
    pdays,
    previous,
    poutcome,
    target_subscribed
FROM bank_marketing
ORDER BY record_id
"""


def clean_pandas(frame: pd.DataFrame) -> pd.DataFrame:
    expected = (
        ["record_id"]
        + FEATURE_COLUMNS
        + ["target_subscribed"]
    )

    missing = set(expected) - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    cleaned = frame.loc[:, expected].copy()

    for name in CATEGORICAL_COLUMNS:
        cleaned[name] = (
            cleaned[name]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        cleaned[name] = cleaned[name].fillna(
            "unknown"
        )

    for name in (
        NUMERIC_COLUMNS
        + ["record_id", "target_subscribed"]
    ):
        cleaned[name] = pd.to_numeric(
            cleaned[name],
            errors="raise",
        )

    if cleaned["record_id"].duplicated().any():
        raise ValueError(
            "Duplicate PostgreSQL record_id values detected"
        )

    if not cleaned["target_subscribed"].isin([0, 1]).all():
        raise ValueError(
            "Unexpected subscription labels"
        )

    if cleaned[NUMERIC_COLUMNS].isna().any().any():
        raise ValueError(
            "Unexpected missing numeric feature values"
        )

    return (
        cleaned
        .sort_values("record_id")
        .reset_index(drop=True)
    )


def clean_with_spark(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, lower, trim

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("BankPredictETL")
        .getOrCreate()
    )

    try:
        sdf = spark.createDataFrame(frame)

        for name in CATEGORICAL_COLUMNS:
            sdf = sdf.withColumn(
                name,
                lower(trim(col(name))),
            )

        return clean_pandas(
            sdf.orderBy("record_id").toPandas()
        )

    finally:
        spark.stop()


def prepare(engine_name: str = "pandas") -> None:
    with database_engine().connect() as connection:
        frame = pd.read_sql_query(
            text(SELECT_SQL),
            connection,
        )

    if frame.empty:
        raise ValueError(
            "PostgreSQL has no data. Run ingestion first."
        )

    if engine_name == "spark":
        cleaned = clean_with_spark(frame)

    elif engine_name == "pandas":
        cleaned = clean_pandas(frame)

    else:
        raise ValueError(
            "engine_name must be pandas or spark"
        )

    PROCESSED_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned.to_csv(
        PROCESSED_CSV,
        index=False,
    )

    print(f"Prepared {len(cleaned):,} rows")

    print(
        f"Number of input features: {len(FEATURE_COLUMNS)}"
    )

    print(
        "Positive class proportion: "
        f"{cleaned['target_subscribed'].mean():.2%}"
    )

    print(f"Saved to {PROCESSED_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--engine",
        choices=["pandas", "spark"],
        default="pandas",
    )

    args = parser.parse_args()

    prepare(args.engine)
