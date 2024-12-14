# Fraudulent Credit Card Transactions Detection

This repository explores methods to detect fraudulent credit card transactions using unsupervised learning.

The dataset used for this project is available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## Objectives

1. Apply Unsupervised algorithms to detect anomalies in credit card transactions.
2. We Compare the performance of:
   - A basic classifier trained in a supervised setting,
   - One-Class SVM, an established method for unsupervised anomaly detection,
   - [Deep SVDD](https://proceedings.mlr.press/v80/ruff18a.html),
   - [DROCC](https://arxiv.org/abs/2002.12718).

---

## Dataset

The dataset used in this project is sourced from the following Kaggle repository:
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  
It contains transactions made by European cardholders over two days, with 492 fraudulent cases out of 284,807 transactions.

---

## TO-DO

1. **Implement Variational Autoencoder (VAE)**:  
   Use reconstruction loss to identify anomalies.
---