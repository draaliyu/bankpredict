
import argparse

import pandas as pd
from sqlalchemy import text

from bankpredict.config import RAW_CSV, database_engine


RENAME = {
    "default": "default_status",
    "duration": "duration_seconds",
    "y": "target_subscribed",
}

EXPECTED = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y",
]


def ingest(replace: bool = False) -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            "Run python -m bankpredict.download first"
        )

    raw = pd.read_csv(
        RAW_CSV,
        sep=";",
    )

    if list(raw.columns) != EXPECTED:
        raise ValueError(
            f"Unexpected data columns: {list(raw.columns)}"
        )

    if not raw["y"].isin(["yes", "no"]).all():
        raise ValueError(
            "Target contains unexpected values"
        )

    raw["y"] = raw["y"].map({
        "yes": 1,
        "no": 0,
    }).astype(int)

    incoming = raw.rename(columns=RENAME)

    engine = database_engine()

    with engine.begin() as connection:
        count = connection.execute(
            text("SELECT COUNT(*) FROM bank_marketing")
        ).scalar_one()

        if count and not replace:
            print(
                f"PostgreSQL already has {count:,} rows."
            )
            print("Use --replace to reload.")
            return

        if replace:
            connection.execute(
                text(
                    "TRUNCATE bank_marketing RESTART IDENTITY"
                )
            )

        incoming.to_sql(
            "bank_marketing",
            connection,
            if_exists="append",
            index=False,
            chunksize=400,
            method="multi",
        )

    print(
        f"Imported {len(incoming):,} rows into PostgreSQL"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--replace",
        action="store_true",
        help="Reload the whole table",
    )

    args = parser.parse_args()

    ingest(replace=args.replace)
