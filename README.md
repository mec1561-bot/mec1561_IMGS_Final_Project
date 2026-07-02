# NASA Software Defect Class Imbalance Project

This project was made for IMGS 789: Machine Learning for Difficult Data.

The goal is to show why accuracy can be misleading when a dataset is imbalanced.
The project uses a NASA software defect dataset from OpenML ID 1067.
The task is to predict whether a software module is defective or non-defective.

## Dataset

OpenML ID: 1067

Original dataset:
- Class 0: 1783 samples
- Class 1: 326 samples
- Imbalance ratio: 5.47:1

Strongly imbalanced dataset:
- Class 0: 1783 samples
- Class 1: 81 samples
- Imbalance ratio: 22.01:1

Class 0 is treated as the majority/non-defective class.
Class 1 is treated as the minority/defective class.

## Models Tested

The project compares these methods:

1. Logistic Regression baseline
2. Random Forest baseline
3. Logistic Regression with class weighting
4. Random Forest with class weighting
5. Logistic Regression with Random Oversampling
6. Logistic Regression with SMOTE

## Metrics Used

The project compares:

- Accuracy
- Balanced accuracy
- Macro-F1
- Precision
- Recall
- ROC-AUC
- AUPRC
- G-mean

Accuracy is included, but the main point is that balanced accuracy, AUPRC, macro-F1, and G-mean are better for imbalanced data.

## Main Result

In the strongly imbalanced dataset, Logistic Regression had:

- Accuracy: 0.957
- Balanced accuracy: 0.500
- AUPRC: 0.107
- G-mean: 0.000

This means the model looked very accurate, but it failed to detect the minority defective class.

After using class weighting, Logistic Regression had:

- Accuracy: 0.714
- Balanced accuracy: 0.711
- AUPRC: 0.111
- G-mean: 0.711

This shows that a lower accuracy model can actually be better when detecting the minority class matters.

## Files

- mec1561_imbalanced_classification_project.py: main Python code
- requirements.txt: packages needed to run the code
- combined_results_table.csv: full results table
- combined_results_summary.txt: text summary of the results
- combined_metric_comparison.png: plot comparing the metrics
- project_report.pdf: final project report
- README.md: project overview and instructions

## How to Run

Create a virtual environment:

py -m venv venv
venv\Scripts\activate

Install packages:

pip install -r requirements.txt

Run the script:

py mec1561_imbalanced_classification_project.py

## Requirements

The project needs:

- numpy
- pandas
- matplotlib
- scikit-learn
- imbalanced-learn

## Conclusion

Accuracy alone is not reliable for imbalanced classification.
In this project, some models had high accuracy but failed on the minority class.
Balanced accuracy, AUPRC, macro-F1, and G-mean gave a better picture of model performance.
