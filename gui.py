"""
Tkinter GUI for the Spam Email Classifier.
"""

from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import joblib

MODEL_PATH = Path("model/spam_classifier.joblib")


class SpamClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Email Classifier - Naive Bayes")
        self.root.geometry("950x700")
        self.root.minsize(750, 550)

        if not MODEL_PATH.exists():
            messagebox.showerror(
                "Model not found",
                "The trained model is missing.\n\n"
                "First run:\n"
                "python download_dataset.py\n"
                "python train.py"
            )
            root.destroy()
            return

        try:
            self.model = joblib.load(MODEL_PATH)
        except Exception as exc:
            messagebox.showerror("Model Error", str(exc))
            root.destroy()
            return

        self.build_ui()

    def build_ui(self):
        main = ttk.Frame(self.root, padding=20)
        main.pack(fill="both", expand=True)

        title = ttk.Label(
            main,
            text="SPAM EMAIL CLASSIFIER",
            font=("Segoe UI", 24, "bold"),
        )
        title.pack(pady=(0, 5))

        subtitle = ttk.Label(
            main,
            text="Multinomial Naive Bayes + TF-IDF | Real SpamAssassin email corpus",
            font=("Segoe UI", 11),
        )
        subtitle.pack(pady=(0, 15))

        ttk.Label(
            main,
            text="Paste an email below:",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")

        text_frame = ttk.Frame(main)
        text_frame.pack(fill="both", expand=True, pady=8)

        self.email_box = tk.Text(
            text_frame,
            wrap="word",
            font=("Consolas", 11),
            undo=True,
            padx=10,
            pady=10,
        )
        scrollbar = ttk.Scrollbar(
            text_frame,
            orient="vertical",
            command=self.email_box.yview,
        )
        self.email_box.configure(yscrollcommand=scrollbar.set)

        self.email_box.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        button_frame = ttk.Frame(main)
        button_frame.pack(pady=12)

        ttk.Button(
            button_frame,
            text="CLASSIFY EMAIL",
            command=self.classify,
        ).pack(side="left", padx=5)

        ttk.Button(
            button_frame,
            text="CLEAR",
            command=self.clear,
        ).pack(side="left", padx=5)

        result_frame = ttk.LabelFrame(
            main,
            text="Classification Result",
            padding=15,
        )
        result_frame.pack(fill="x", pady=5)

        self.result_label = ttk.Label(
            result_frame,
            text="Enter an email and click CLASSIFY EMAIL",
            font=("Segoe UI", 18, "bold"),
        )
        self.result_label.pack(pady=5)

        self.spam_var = tk.StringVar(value="Spam probability: --")
        self.ham_var = tk.StringVar(value="No-spam probability: --")

        ttk.Label(
            result_frame,
            textvariable=self.spam_var,
            font=("Segoe UI", 11),
        ).pack()

        ttk.Label(
            result_frame,
            textvariable=self.ham_var,
            font=("Segoe UI", 11),
        ).pack()

        ttk.Label(
            main,
            text="Model: TF-IDF → Multinomial Naive Bayes",
            font=("Segoe UI", 9),
        ).pack(pady=(10, 0))

    def classify(self):
        email_text = self.email_box.get("1.0", "end").strip()

        if not email_text:
            messagebox.showwarning(
                "No Email",
                "Please paste an email before classifying.",
            )
            return

        try:
            prediction = self.model.predict([email_text])[0]
            probabilities = self.model.predict_proba([email_text])[0]
            classes = list(self.model.classes_)
            probability_map = dict(zip(classes, probabilities))

            spam_probability = probability_map.get("spam", 0.0)
            ham_probability = probability_map.get("ham", 0.0)

            if prediction == "spam":
                self.result_label.configure(
                    text="🚨 SPAM EMAIL",
                    foreground="#b00020",
                )
            else:
                self.result_label.configure(
                    text="✓ NO SPAM / HAM",
                    foreground="#087f23",
                )

            self.spam_var.set(
                f"Spam probability: {spam_probability * 100:.2f}%"
            )
            self.ham_var.set(
                f"No-spam probability: {ham_probability * 100:.2f}%"
            )

        except Exception as exc:
            messagebox.showerror("Prediction Error", str(exc))

    def clear(self):
        self.email_box.delete("1.0", "end")
        self.result_label.configure(
            text="Enter an email and click CLASSIFY EMAIL",
            foreground="",
        )
        self.spam_var.set("Spam probability: --")
        self.ham_var.set("No-spam probability: --")


def main():
    root = tk.Tk()
    SpamClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
