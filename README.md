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

## Models Used
1. Logistic Regression (Primary)
   - Accuracy: 97.49%
   - ROC-AUC: 0.9934

2. Naive Bayes (Baseline)
   - Accuracy: 97.10%
   - ROC-AUC: 0.9904

## Usage
1. Install dependencies: `pip install -r requirements.txt` (if provided) or install `pandas`, `scikit-learn`, and other listed libraries.
2. Prepare the dataset (download from Kaggle and place under `data/`), then run the training script: `python train.py`.
3. Use the provided inference script or notebook to test single messages.

## License & Contributing
Feel free to open issues or pull requests for improvements. Include reproducible steps and tests when proposing changes.
