"""
Train the spam email classifier.

Pipeline:
Raw email text -> TF-IDF -> Multinomial Naive Bayes

The complete sklearn Pipeline is saved, so the same preprocessing used during
training is automatically applied when predicting a new email.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

DATA_PATH = Path("dataset/processed/emails.csv")
MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")
MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"
METRICS_PATH = RESULTS_DIR / "metrics.json"
REPORT_PATH = RESULTS_DIR / "classification_report.txt"
CM_PATH = RESULTS_DIR / "confusion_matrix.png"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run: python download_dataset.py"
        )

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH).fillna("")
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                max_features=100000,
            ),
        ),
        (
            "classifier",
            MultinomialNB(alpha=0.1),
        ),
    ])

    print("Training Multinomial Naive Bayes...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label="spam")
    recall = recall_score(y_test, predictions, pos_label="spam")
    f1 = f1_score(y_test, predictions, pos_label="spam")

    report = classification_report(y_test, predictions, digits=4)

    metrics = {
        "dataset_size": int(len(df)),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "ham_samples": int((y == "ham").sum()),
        "spam_samples": int((y == "spam").sum()),
        "accuracy": round(float(accuracy), 6),
        "spam_precision": round(float(precision), 6),
        "spam_recall": round(float(recall), 6),
        "spam_f1": round(float(f1), 6),
        "model": "Multinomial Naive Bayes",
        "vectorizer": "TF-IDF",
        "test_size": 0.20,
        "random_state": 42,
    }

    joblib.dump(pipeline, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Spam", "Spam"],
    )
    disp.plot()
    plt.title("Spam Email Classifier - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()

    print("\n===== RESULTS =====")
    print(f"Dataset size : {len(df)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples : {len(X_test)}")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1-score     : {f1:.4f}")
    print("\nClassification report:\n")
    print(report)

    print("Saved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved report:", REPORT_PATH)
    print("Saved confusion matrix:", CM_PATH)


if __name__ == "__main__":
    main()
