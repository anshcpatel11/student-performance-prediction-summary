# Student Performance Prediction — Machine Learning Project

This project explores academic performance prediction using classical machine learning techniques and custom-built algorithms. The goal is to predict whether a student is at academic risk (defined as earning a final grade below 10) using demographic, behavioral, and school-related features from the UCI Student Performance dataset.

The project demonstrates end-to-end ML workflow: data cleaning, feature engineering, algorithm design, model evaluation, and experimentation with ensemble methods.

## Disclaimer
This repository contains a high-level overview of my project work for portfolio purposes.  
To protect academic integrity and prevent reuse in coursework, only selected files are publicly visible.  
If you would like to review the full source code for professional or recruiting purposes, feel free to contact me and I can provide access.


## Project Highlights

### **Phase 1 — Core ML Pipeline + From-Scratch Algorithms**

* Robust data loading that handles multiple dataset formats
* Comprehensive preprocessing:

  * Target generation (`at_risk`)
  * Categorical encoding
  * Median imputation
  * Min-max scaling
* Exploratory statistics + correlation analysis helpers
* Baseline model using Gradient Boosting
* **Custom implementations (NumPy-only):**

  * Decision Tree classifier
  * Random Forest with bootstrap sampling and majority voting

### **Phase 2 — Modeling Enhancements**

* Transitioned to full dataset (395 rows)
* Feature selection using ANOVA F-scores (top-K)
* Stacking ensemble combining:

  * Gradient Boosting
  * Random Forest
  * Logistic Regression meta-model

---

##  **Dataset**

Uses the **UCI Student Performance Dataset**, which includes:

* Demographics
* Family background
* Study habits
* School support indicators
* Prior grades

Target variable:

```
at_risk = 1 if G3 < 10 else 0
```

Dataset reference:
[https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)

---

## **Technologies Used**

* Python
* NumPy
* Pandas
* scikit-learn
* Custom algorithm development
* Data preprocessing and feature engineering

---

## **Project Structure**

```
src/
    student_project.py     # preprocessing + custom tree + custom random forest
    phase2.py              # experiments and modeling enhancements

data/
    student-mat.csv

reports/
    Phase2_Report.pdf

requirements.txt
README.md
```

This structure supports adding new ML projects in the future under the same repo.

---

## **Model Performance**

### **Baseline Gradient Boosting (Full Dataset)**

| Metric   | Score |
| -------- | ----- |
| Accuracy | 0.681 |
| F1 Score | 0.406 |
| ROC AUC  | 0.675 |

### **Gradient Boosting + Feature Selection (Top 20 Features)**

| Metric   | Score |
| -------- | ----- |
| Accuracy | 0.739 |
| F1 Score | 0.523 |
| ROC AUC  | 0.727 |

### **Stacking Ensemble**

| Metric   | Score |
| -------- | ----- |
| Accuracy | 0.723 |
| F1 Score | 0.377 |
| ROC AUC  | 0.686 |

Feature selection produced the most balanced and performant model overall.

---

## **How to Run the Project**

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all Phase 2 experiments:

```bash
python -m src.phase2
```


## **Author**

**Ansh Patel**
GitHub: [https://github.com/anshcpatel11](https://github.com/anshcpatel11)
