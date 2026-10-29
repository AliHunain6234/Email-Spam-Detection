# Email Spam Detection

A machine-learning project that classifies short text messages as **spam** or **not spam (ham)** with TF-IDF features and scikit-learn classifiers. It contains two notebooks: a simple, carefully evaluated **Naive Bayes baseline**, and a larger experiment with **text cleaning, extra features, class balancing, four models and an ensemble**. It was built as a Month 1 task of an AI/ML internship.

## The data

`spam.csv` is the classic **SMS Spam Collection**: 5,572 text messages labelled `ham` (a normal message) or `spam`. Despite the project's name, the messages are short SMS texts, not full emails, so results on real emails may differ.

| | Messages |
|---|---|
| Original file | 5,572 (4,825 ham, 747 spam) |
| After removing 403 exact duplicates | **5,169** (4,516 ham, 653 spam, about 12.6% spam) |

## Notebook 1: Naive Bayes baseline (`Email_Spam_Classification.ipynb`)

1. Loads the data, drops three empty columns, removes the 403 duplicates and encodes the labels.
2. Adds simple text statistics (characters, words, sentences) and plots the class balance and a pair plot.
3. Turns the lower-cased messages into TF-IDF vectors (3,000 features).
4. Splits 80/20 (4,135 train, 1,034 test) and compares three Naive Bayes models on the test set:

| Model | Accuracy | Spam wrongly passed (missed) | Normal messages wrongly flagged |
|---|---|---|---|
| Bernoulli NB | **98.45%** | 13 of 138 | 3 of 896 |
| Multinomial NB | 97.29% | 28 of 138 | **0 of 896** |
| Gaussian NB | 89.85% | 21 of 138 | 84 of 896 |

Context for the numbers: 86.7% of the test messages are ham, so a model that always answered "not spam" would already score 86.7%. Multinomial NB never flags a normal message, which is useful when a false alarm is costly; Bernoulli NB catches more spam.

## Notebook 2: larger experiment (`Spam_Email_Detection.ipynb`)

1. Same cleaning, then a text-preprocessing function (lower-casing, tokenising, keeping alphanumeric words, removing English stop-words, Porter stemming).
2. Extra features: number of characters, number of words and number of special characters (`! @ # $ % ^ & *`), stacked next to the 3,000 TF-IDF features.
3. Balances the classes with **SMOTE** (synthetic spam examples), then splits 80/20.
4. Trains Naive Bayes, Logistic Regression, Random Forest and an SVM, and a **soft-voting ensemble** of Logistic Regression, Naive Bayes and Random Forest:

| Model | Test accuracy |
|---|---|
| Naive Bayes | 0.98 |
| Logistic Regression | 0.98 |
| Random Forest | 0.99 |
| SVM (default settings) | 0.86 |
| **Ensemble (LR + NB + RF)** | **0.99** |

5. Saves the vectorizer and the ensemble with `pickle` (`vectorizer.pkl`, `spam_model.pkl`, not stored in this repository) and defines `predict_spam(text)`, which returns a label and the spam probability. For example, "Congratulations! You won $1000 cash prize!" is classified as spam (probability about 0.75), and "I live in Pakistan...!" as not spam (spam probability about 0.31).

**Read these scores with care.** SMOTE is applied to the whole dataset *before* the train/test split, so synthetic messages made from the same real ones end up on both sides of the split, and the test set is balanced (about half spam) unlike real life. The 0.99 is therefore optimistic. The numbers in Notebook 1 are measured on a realistic test set and are the more trustworthy ones.

## Run it

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open one of the notebooks and run it from top to bottom. Keep `spam.csv` in the same folder. The first run downloads the NLTK data (`punkt`, `stopwords`) the notebooks need.

## Limitations

- Short SMS text only; no email headers, links or attachments are used.
- In Notebook 1 the stemming and stop-word function is defined but then replaced by a version that only lower-cases the text, so the baseline uses lower-cased words without stemming.
- No cross-validation or hyperparameter tuning; each result comes from one split.
- The SVM in Notebook 2 uses default settings on unscaled features, which probably explains its lower score.
- A spam filter that hides messages should keep false alarms very low; check the false-alarm counts, not only the accuracy.
