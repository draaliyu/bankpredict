
import io
import zipfile

import pandas as pd
import requests

from bankpredict.config import RAW_CSV


DATASET_URLS = [
    "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
]

REQUIRED_COLUMNS = {
    "age",
    "job",
    "balance",
    "duration",
    "campaign",
    "y",
}


def read_bank_csv_from_zip(data: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.namelist()

        direct = next(
            (
                name
                for name in members
                if name.lower().endswith("bank-full.csv")
            ),
            None,
        )

        if direct:
            return archive.read(direct)

        nested = next(
            (
                name
                for name in members
                if name.lower().endswith("bank.zip")
            ),
            None,
        )

        if nested:
            return read_bank_csv_from_zip(archive.read(nested))

    raise ValueError(
        "The downloaded archive does not contain bank-full.csv"
    )


def download_dataset() -> None:
    if RAW_CSV.exists():
        print(f"Dataset already present: {RAW_CSV}")
        return

    errors = []

    for url in DATASET_URLS:
        try:
            print(f"Downloading source data: {url}")

            response = requests.get(url, timeout=90)
            response.raise_for_status()

            csv_bytes = read_bank_csv_from_zip(
                response.content
            )

            preview = pd.read_csv(
                io.BytesIO(csv_bytes),
                sep=";",
            )

            if not REQUIRED_COLUMNS.issubset(preview.columns):
                raise ValueError("Missing required dataset columns")

            if len(preview) < 1000:
                raise ValueError("Dataset is unexpectedly small")

            RAW_CSV.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            RAW_CSV.write_bytes(csv_bytes)

            print(
                f"Saved {len(preview):,} records to {RAW_CSV}"
            )

            return

        except (
            requests.RequestException,
            zipfile.BadZipFile,
            ValueError,
        ) as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Unable to download the dataset.\n"
        + "\n".join(errors)
    )


if __name__ == "__main__":
    download_dataset()
