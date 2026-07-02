# Comparing Performance Metrics Under Class Imbalance for NASA Software Defect Prediction

## Project Overview

This project was completed for **IMGS 789: Machine Learning for Difficult Data**. The goal of the project is to study how class imbalance affects machine learning evaluation metrics using a NASA software defect prediction dataset.

The task is a binary classification problem: predicting whether a software module is defective or non-defective using software metric features. This is a useful example of imbalanced learning because defective software modules are usually less common than non-defective modules, but they are often the more important class to detect.

The main research question is:

> When does accuracy become misleading under class imbalance, and which metrics better reflect model behavior?

## Dataset

The dataset used in this project is a NASA software defect prediction dataset from **OpenML ID 1067**.

- Number of samples: 2109
- Number of features: 21
- Task: Binary classification
- Class 0: Majority class, treated as likely non-defective
- Class 1: Minority class, treated as likely defective

Original class distribution:

| Class | Count |
|---|---:|
| 0 | 1783 |
| 1 | 326 |

Original imbalance ratio:

```text
1783:326 ≈ 5.47:1
