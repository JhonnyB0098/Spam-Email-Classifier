"""
Command-line email spam prediction.
"""

from pathlib import Path
import sys
import joblib

MODEL_PATH = Path("model/spam_classifier.joblib")


def predict_email(email_text: str):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python train.py"
        )

    model = joblib.load(MODEL_PATH)
    prediction = model.predict([email_text])[0]
    probabilities = model.predict_proba([email_text])[0]
    classes = list(model.classes_)

    probability_map = dict(zip(classes, probabilities))

    return prediction, probability_map


def main():
    if len(sys.argv) > 1:
        email_text = " ".join(sys.argv[1:])
    else:
        print("Paste the email. Finish with an empty line:")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        email_text = "\n".join(lines)

    prediction, probabilities = predict_email(email_text)

    print("\n===== PREDICTION =====")
    print("Result:", "SPAM" if prediction == "spam" else "NO SPAM")
    print(f"Spam probability   : {probabilities.get('spam', 0) * 100:.2f}%")
    print(f"Ham probability    : {probabilities.get('ham', 0) * 100:.2f}%")


if __name__ == "__main__":
    main()
