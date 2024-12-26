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

| Metric               | Basic NN Supervised | Gradient Boosting | Deep SVDD | DROCC  | VAE    |
|----------------------|---------------------|-------------------|-----------|--------|--------|
| **ROC-AUC Scores**    | 0.942619            | 0.969196          | 0.946778  | 0.955337 | 0.922005 |
| **F1 Scores**         | 0.994224            | 0.995913          | 0.996056  | 0.996430 | 0.994965 |
| **Accuracy**          | 0.988526            | 0.991868          | 0.992149  | 0.992893 | 0.989986 |
| **Precision**         | 0.999716            | 0.999773          | 0.999561  | 0.999689 | 0.999319 |
| **Recall**            | 0.988792            | 0.992082          | 0.992575  | 0.993193 | 0.990648 |


---

## TO-DO

Streamlit interface to play with threshold.

---
