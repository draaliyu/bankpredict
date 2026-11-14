
import hashlib
import json

from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from bankpredict.config import (
    METADATA_FILE,
    MODEL_FILE,
    PROCESSED_CSV,
    REPORTS_DIR,
)

from bankpredict.model import build_model
from bankpredict.schema import FEATURE_COLUMNS


RANDOM_SEED = 42


def summarize(y_true, probabilities, threshold):
    labels = (
        probabilities >= threshold
    ).astype(int)

    return {
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "precision": float(
            precision_score(
                y_true,
                labels,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                labels,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                labels,
                zero_division=0,
            )
        ),
    }


def select_threshold(y_true, probabilities):
    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    scores = (
        2
        * precision[:-1]
        * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_index = int(np.argmax(scores))

    return float(thresholds[best_index])


def top_decile_metrics(y_true, probabilities):
    ranked = np.argsort(-probabilities)

    target = max(
        1,
        int(np.ceil(len(probabilities) * 0.10)),
    )

    chosen = np.asarray(y_true)[
        ranked[:target]
    ]

    baseline = float(np.mean(y_true))

    precision_at_10 = float(
        np.mean(chosen)
    )

    captured = float(
        chosen.sum()
        / max(1, np.sum(y_true))
    )

    return {
        "contact_budget_fraction": 0.10,
        "contacted_customers": int(target),
        "precision_at_10_percent": precision_at_10,
        "lift_at_10_percent": (
            float(precision_at_10 / baseline)
            if baseline
            else 0.0
        ),
        "captured_subscribers_fraction": captured,
        "test_prevalence": baseline,
    }


def save_precision_recall_plot(
    y_true,
    probabilities,
):
    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        recall,
        precision,
        label="Selected model",
    )

    plt.axhline(
        float(np.mean(y_true)),
        linestyle="--",
        label="Test prevalence",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "test_precision_recall.png",
        dpi=180,
    )

    plt.close()


def save_confusion_plot(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)

    plt.xticks([0, 1], ["No", "Yes"])
    plt.yticks([0, 1], ["No", "Yes"])

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Test Confusion Matrix")

    for row in range(2):
        for column in range(2):
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="black",
            )

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "test_confusion_matrix.png",
        dpi=180,
    )

    plt.close()


def train():
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(
            "Run python -m bankpredict.prepare first"
        )

    frame = (
        pd.read_csv(PROCESSED_CSV)
        .sort_values("record_id")
        .reset_index(drop=True)
    )

    if len(frame) < 1000:
        raise ValueError(
            "Unexpectedly small dataset"
        )

    n = len(frame)

    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)

    train_frame = frame.iloc[:train_end]
    validation_frame = frame.iloc[
        train_end:validation_end
    ]
    test_frame = frame.iloc[validation_end:]

    x_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame[
        "target_subscribed"
    ].astype(int)

    x_val = validation_frame[FEATURE_COLUMNS]
    y_val = validation_frame[
        "target_subscribed"
    ].astype(int)

    x_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame[
        "target_subscribed"
    ].astype(int)

    partitions = [
        ("train", y_train),
        ("validation", y_val),
        ("test", y_test),
    ]

    for name, labels in partitions:
        if labels.nunique() != 2:
            raise ValueError(
                f"{name} partition has just one class"
            )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        "BankPredict"
    )

    experiments = {}

    data_hash = hashlib.sha256(
        PROCESSED_CSV.read_bytes()
    ).hexdigest()

    for model_name in [
        "logistic_regression",
        "xgboost",
    ]:
        print(
            f"Training {model_name} ..."
        )

        model = build_model(model_name)

        with mlflow.start_run(
            run_name=f"candidate-{model_name}"
        ):
            mlflow.set_tags({
                "stage": "validation",
                "dataset_sha256": data_hash,
            })

            mlflow.log_params({
                "model_name": model_name,
                "seed": RANDOM_SEED,
                "split": "ordered_60_20_20",
            })

            model.fit(
                x_train,
                y_train,
            )

            val_prob = model.predict_proba(
                x_val
            )[:, 1]

            val_metrics = summarize(
                y_val,
                val_prob,
                threshold=0.5,
            )

            mlflow.log_metrics({
                f"val_{name}": value
                for name, value
                in val_metrics.items()
            })

        experiments[model_name] = {
            "model": model,
            "val_prob": val_prob,
            "val_metrics": val_metrics,
        }

        print(
            "Validation average precision: "
            f"{val_metrics['average_precision']:.4f}"
        )

    winner_name = max(
        experiments,
        key=lambda name: (
            experiments[name]["val_metrics"][
                "average_precision"
            ]
        ),
    )

    winner = experiments[
        winner_name
    ]["model"]

    threshold = select_threshold(
        y_val,
        experiments[
            winner_name
        ]["val_prob"],
    )

    test_prob = winner.predict_proba(
        x_test
    )[:, 1]

    test_metrics = summarize(
        y_test,
        test_prob,
        threshold,
    )

    campaign_metrics = top_decile_metrics(
        y_test.to_numpy(),
        test_prob,
    )

    metadata = {
        "model_name": winner_name,
        "model_version": "1.0.0",
        "created_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "decision_threshold": threshold,
        "features": FEATURE_COLUMNS,
        "dataset_sha256": data_hash,
        "validation_metrics": (
            experiments[
                winner_name
            ]["val_metrics"]
        ),
        "test_metrics": test_metrics,
        "campaign_metrics": campaign_metrics,
        "train_count": len(x_train),
        "validation_count": len(x_val),
        "test_count": len(x_test),
        "holdout_note": (
            "Source-order holdout; "
            "no true dates or customer IDs."
        ),
    }

    joblib.dump(
        winner,
        MODEL_FILE,
    )

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        REPORTS_DIR / "metrics.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    predictions = (
        test_prob >= threshold
    ).astype(int)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    (
        REPORTS_DIR
        / "test_classification_report.txt"
    ).write_text(
        report,
        encoding="utf-8",
    )

    pd.DataFrame({
        "record_id": test_frame[
            "record_id"
        ].to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted_probability": test_prob,
        "selected": predictions,
    }).to_csv(
        REPORTS_DIR / "test_predictions.csv",
        index=False,
    )

    save_precision_recall_plot(
        y_test,
        test_prob,
    )

    save_confusion_plot(
        y_test,
        test_prob,
        threshold,
    )

    with mlflow.start_run(
        run_name=(
            f"selected-{winner_name}-holdout"
        )
    ):
        mlflow.set_tags({
            "stage": "final_holdout",
            "model_name": winner_name,
            "dataset_sha256": data_hash,
        })

        mlflow.log_param(
            "decision_threshold",
            threshold,
        )

        mlflow.log_metrics({
            f"test_{name}": value
            for name, value
            in test_metrics.items()
        })

        mlflow.log_metrics({
            name: float(value)
            for name, value
            in campaign_metrics.items()
        })

        mlflow.log_artifact(
            str(MODEL_FILE),
            artifact_path="model",
        )

        mlflow.log_artifact(
            str(METADATA_FILE),
            artifact_path="model",
        )

        mlflow.log_artifacts(
            str(REPORTS_DIR),
            artifact_path="reports",
        )

    print(
        f"Selected model: {winner_name}"
    )

    print(
        "Decision threshold: "
        f"{threshold:.3f}"
    )

    print(
        "Test average precision: "
        f"{test_metrics['average_precision']:.4f}"
    )

    print(
        "Top 10% contact-list lift: "
        f"{campaign_metrics['lift_at_10_percent']:.2f}x"
    )

    print(
        f"Saved model: {MODEL_FILE}"
    )


if __name__ == "__main__":
    train()
