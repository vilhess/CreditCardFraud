# Fraudulent Credit Card Transactions Detection

This repository explores methods to detect fraudulent credit card transactions. Specifically, we implement and evaluate the [DROCC algorithm](https://arxiv.org/abs/2002.12718) in an **unsupervised learning setup**. 

The dataset used for this project is available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## Objectives

1. Apply the DROCC algorithm to detect anomalies in credit card transactions.
2. Compare the performance of DROCC against:
   - A basic classifier trained in a supervised setting.
   - One-Class SVM, an established method for unsupervised anomaly detection.

3. Extend the scope to include additional unsupervised methods (listed in the TO-DO section).

---

## Dataset

The dataset used in this project is sourced from the following Kaggle repository:
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  
It contains transactions made by European cardholders over two days, with 492 fraudulent cases out of 284,807 transactions.

---

## Methodologies

1. **DROCC Algorithm**  
   A state-of-the-art approach for unsupervised anomaly detection. DROCC is designed to be robust in detecting outliers within high-dimensional data.

2. **Baseline Methods**  
   - **Supervised Classifier**: A standard supervised learning model to establish a baseline.  
   - **One-Class SVM**: A traditional anomaly detection method that isolates outliers based on kernelized hyperplanes.

---

## TO-DO

1. **Implement Variational Autoencoder (VAE)**:  
   Use reconstruction loss to identify anomalies.

2. **Deep SVDD**:  
   Explore this deep learning method for one-class classification tasks.

---