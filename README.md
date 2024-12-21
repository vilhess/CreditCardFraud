# Fraudulent Credit Card Transactions Detection

This repository explores methods to detect fraudulent credit card transactions using unsupervised learning.

The dataset used for this project is available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## Objectives

1. Apply Unsupervised algorithms to detect anomalies in credit card transactions.
2. We Compare the performance of:
   - A basic classifier trained in a supervised setting,
   - A Gradient-Boosting classifier in a supervised setting,
   - [Deep SVDD](https://proceedings.mlr.press/v80/ruff18a.html),
   - [DROCC](https://arxiv.org/abs/2002.12718).
   - A VAE using as a score the reconstruction loss.
3. Implementation of test p-values using a validation set to reject anomalous samples, based on the method proposed in the paper: [Testing for Outliers with Conformal p-values](https://arxiv.org/abs/2104.08279).
4. Performance is measured using the AUC-ROC score, F1 score, recall, precision, and accuracy.

---

## Dataset

The dataset used in this project is sourced from the following Kaggle repository:
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  
It contains transactions made by European cardholders over two days, with 492 fraudulent cases out of 284,807 transactions.

---

## Results

| Metric         | Basic NN Supervised | One-Class SVM SGD | Deep SVDD | DROCC    | VAE      |
|----------------|----------------------|-------------------|-----------|----------|----------|
| ROC-AUC Scores | 0.937028            | 0.505303          | 0.950257  | 0.954082 | 0.931480 |
| F1 Scores      | 0.993953            | 0.002650          | 0.995779  | 0.996367 | 0.994780 |
| Accuracy       | 0.987992            | 0.016755          | 0.991601  | 0.992767 | 0.989621 |
| Precision      | 0.999701            | 0.001327          | 0.999561  | 0.999689 | 0.999305 |
| Recall         | 0.988271            | 0.989362          | 0.992026  | 0.993067 | 0.990296 |


---

## TO-DO

Streamlit interface to play with threshold.
Furthermore, implement a Gradient Boosting algorithms under the supervised setting as a baseline

---
