# Spam Email Classifier

A complete machine-learning project for classifying emails as **SPAM** or **NO SPAM (HAM)** using:

- Real email data from the Apache SpamAssassin Public Corpus
- Python
- TF-IDF text vectorization
- Multinomial Naive Bayes
- Scikit-learn
- Tkinter GUI

## 1. Project workflow

```text
SpamAssassin real email corpus
             |
             v
      Download archives
             |
             v
       Parse raw emails
             |
             v
       Clean email text
             |
             v
         TF-IDF
             |
             v
   Multinomial Naive Bayes
             |
             v
       Train / Test
             |
             v
     Evaluation metrics
             |
             v
      Saved ML pipeline
             |
             v
         Tkinter GUI
```

## 2. Dataset

This project uses the real Apache SpamAssassin Public Mail Corpus.

The downloader uses these public 2003 archives:

- 20030228_easy_ham.tar.bz2
- 20030228_easy_ham_2.tar.bz2
- 20030228_hard_ham.tar.bz2
- 20030228_spam.tar.bz2
- 20030228_spam_2.tar.bz2

The dataset contains real email messages labeled as spam or non-spam. No synthetic emails are generated.

The raw archives are downloaded automatically by `download_dataset.py`, so they are not bundled into this project.

## 3. Requirements

Python 3.10 or newer is recommended.

Install packages:

```bash
pip install -r requirements.txt
```

Tkinter is normally included with Python on Windows.

## 4. Recommended virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 5. Download and prepare the real dataset

Run:

```bash
python download_dataset.py
```

This will:

1. Download the real SpamAssassin archives.
2. Extract the archives.
3. Parse the raw RFC email files.
4. Extract useful subject/body text.
5. Ignore file attachments.
6. Remove exact duplicate messages.
7. Create:

```text
dataset/processed/emails.csv
```

## 6. Train the Naive Bayes classifier

Run:

```bash
python train.py
```

The script:

- loads the real email dataset
- creates an 80/20 stratified train/test split
- converts text to TF-IDF features
- trains Multinomial Naive Bayes
- evaluates the classifier
- saves the complete pipeline

Output files:

```text
model/spam_classifier.joblib
results/metrics.json
results/classification_report.txt
results/confusion_matrix.png
```

## 7. Test from command line

Example:

```bash
python predict.py "Congratulations! You have won a cash prize. Click here to claim your reward now."
```

Or run:

```bash
python predict.py
```

and paste a complete email.

## 8. Start the GUI

After training:

```bash
python gui.py
```

The GUI allows you to paste an email and receive:

- SPAM / NO SPAM result
- Spam probability
- Ham probability

## 9. Why Naive Bayes?

Multinomial Naive Bayes is a strong baseline for text classification because it works efficiently with high-dimensional word features.

The classifier estimates:

```text
P(class | email text)
```

using the probabilities of the words/features observed in the email.

For example, if an email contains many patterns commonly associated with spam, its estimated spam probability increases.

## 10. Why TF-IDF?

TF-IDF converts email text into numerical features.

A word gets a higher weight when it is important to a particular document but is not extremely common across the entire training collection.

This allows the Naive Bayes classifier to work with numerical representations of the email text.

## 11. Model architecture

```text
Input email
    |
    v
TF-IDF Vectorizer
    |
    v
MultinomialNB(alpha=0.1)
    |
    v
Spam / Ham probability
    |
    v
Final classification
```

The saved `.joblib` file contains the complete sklearn pipeline, including the vectorizer and classifier.

## 12. Evaluation

The training script reports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

For spam detection, precision and recall are particularly important.

### Precision

Of all emails predicted as spam, how many were actually spam?

### Recall

Of all actual spam emails, how many did the model detect?

### F1-score

Harmonic mean of precision and recall.

## 13. Project limitations

This is an educational machine-learning classifier, not a production email security gateway.

The SpamAssassin corpus is an older public corpus, so modern phishing/spam campaigns may use language and URLs that differ from the training data.

The model should therefore be retrained with newer, representative data before being used for real-world filtering.

## 14. Suggested viva explanation

**Question: What is the project?**

This project is a machine-learning based spam email classifier. It uses TF-IDF to convert email text into numerical features and Multinomial Naive Bayes to classify an email as spam or ham.

**Question: What dataset did you use?**

The project uses the Apache SpamAssassin Public Mail Corpus, which contains real email messages labeled as spam and non-spam.

**Question: Why Naive Bayes?**

Naive Bayes is computationally efficient and performs well for text classification problems with many word features.

**Question: What is TF-IDF?**

TF-IDF stands for Term Frequency-Inverse Document Frequency. It represents the importance of a word in a document relative to the complete collection of documents.

**Question: What is ham?**

Ham means a legitimate, non-spam email.

**Question: What is the input and output?**

Input is the text of an email. Output is either SPAM or NO SPAM, together with the estimated probabilities.

## 15. Data source

Apache SpamAssassin Public Corpus:

https://spamassassin.apache.org/old/publiccorpus/

The project downloads the data automatically, so manual dataset preparation is not required.
