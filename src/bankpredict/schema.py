
FEATURE_COLUMNS = [
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
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]

NUMERIC_COLUMNS = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column not in NUMERIC_COLUMNS
]
