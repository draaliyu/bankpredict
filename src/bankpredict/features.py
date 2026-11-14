
import numpy as np
import pandas as pd


def add_contact_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    enriched = frame.copy()

    pdays = pd.to_numeric(
        enriched["pdays"],
        errors="coerce",
    )

    enriched["previously_contacted"] = (
        pdays >= 0
    ).astype(int)

    enriched["days_since_previous_contact"] = (
        pdays.where(
            pdays >= 0,
            np.nan,
        )
    )

    return enriched
