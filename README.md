# SMS Spam Detection

## Description
This repository implements an SMS spam detection system using machine learning. It demonstrates end-to-end text classification on the SMS Spam Collection dataset from Kaggle, transforming raw SMS messages into features and training supervised classifiers to distinguish "ham" (not spam) from "spam".

The code is intended as a compact, reproducible example using standard Python libraries (pandas, scikit-learn) and includes preprocessing (cleaning, tokenization), feature extraction (TF-IDF/CountVectorizer), model training, evaluation, and simple model comparison.

## Key Goals
- Provide a clear, reproducible pipeline for preprocessing SMS text data.
- Compare baseline (Naive Bayes) and primary (Logistic Regression) classifiers.
- Report standard evaluation metrics (accuracy, ROC-AUC) and provide examples for inference on new messages.

## Overview
SMS Spam Detection using Machine Learning (Logistic Regression)

## Dataset
- Source: Kaggle - SMS Spam Collection Dataset
- Total Messages: ~5,500
- Classes: Ham (Not Spam) & Spam

## Extracted Features
The preprocessing and feature-extraction stage produces the following features (common choices used in this project):
- TF-IDF vectors of tokenized text (unigrams and optional bigrams)
- Term/word counts (CountVectorizer)
- Message length (number of characters and number of tokens)
- Number of punctuation characters and special symbols
- Presence of URLs, email addresses, or phone numbers
- Ratio of uppercase characters (capitalization signal)
- Count of numeric tokens
- Stopword ratio (stopwords / total tokens)
- Optional n-gram features (bigrams/trigrams) for phrase signals

These feature sets can be combined or selected depending on the classifier and validation results.

## Models Used
1. Logistic Regression (Primary)
   - Accuracy: 97.49%
   - ROC-AUC: 0.9934

2. Naive Bayes (Baseline)
   - Accuracy: 97.10%
   - ROC-AUC: 0.9904

## Classifier Comparison
| Classifier           | Accuracy | ROC-AUC | Notes / Which is better |
|----------------------|---------:|--------:|:------------------------|
| Logistic Regression  |   97.49% |  0.9934 | Primary model — best overall performance in this evaluation |
| Naive Bayes          |   97.10% |  0.9904 | Strong baseline; slightly lower than Logistic Regression |

Based on the measured metrics (accuracy and ROC-AUC), Logistic Regression performs marginally better and is chosen as the primary model for this project.

## Outcome
The models trained on the SMS Spam Collection dataset achieve high performance (≈97% accuracy and >0.99 ROC-AUC), indicating that simple text representations (TF-IDF or counts) combined with classic classifiers are effective for this task. Logistic Regression provides the best trade-off between simplicity, interpretability, and performance here. However, differences are small — Naive Bayes remains a competitive baseline for quick prototyping and low-resource environments.

Practical notes:
- Evaluate models on a held-out test set and inspect precision/recall for the spam class when deploying (spam detection often requires high precision to avoid false positives).
- Consider threshold tuning and cost-sensitive learning if false positives/negatives have asymmetric costs.

## Usage
1. Install dependencies: `pip install -r requirements.txt` (if provided) or install `pandas`, `scikit-learn`, and other listed libraries.
2. Prepare the dataset (download from Kaggle and place under `data/`), then run the training script: `python train.py`.
3. Use the provided inference script or notebook to test single messages.

## License & Contributing
Feel free to open issues or pull requests for improvements. Include reproducible steps and tests when proposing changes.
