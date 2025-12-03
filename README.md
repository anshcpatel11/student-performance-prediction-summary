# Student Performance Prediction (Machine Learning Project)

This repository contains a machine learning project focused on predicting whether a student is academically at risk, defined as earning a final grade (G3) below 10. The work includes end‑to‑end preprocessing, classical ML models, and a custom Random Forest built entirely from scratch using NumPy.

## Overview

The project is structured in two phases:

### Phase 1
- Implement robust data loading
- Build preprocessing pipeline:
  - Target creation (`at_risk`)
  - One‑hot encoding
  - Median imputation
  - Min‑max scaling
- Add exploratory helpers (summary stats, correlations)
- Implement baseline Gradient Boosting model
- Build a **custom Decision Tree and Random Forest** without scikit‑learn

### Phase 2
- Use the full dataset (395 rows)
- Perform feature selection (top‑K using F‑scores)
- Build a stacking ensemble combining:
  - Gradient Boosting
  - Random Forest
  - Logistic Regression (meta‑learner)

## Dataset

The project uses the **UCI Student Performance Dataset**, including student background, behavior, family attributes, and prior grades.

Target variable:

```
at_risk = 1 if G3 < 10 else 0
```

Dataset reference:  
https://archive.ics.uci.edu/dataset/320/student+performance

## Technologies Used
- Python
- NumPy
- Pandas
- scikit‑learn
- Custom algorithm implementation

## Project Structure

```
src/
    student_project.py
    phase2.py

data/
    student-mat.csv

reports/
    Phase2_Report.pdf

requirements.txt
README.md
```

## Modeling Summary

### Baseline Gradient Boosting (Full Dataset)
- Accuracy: 0.681  
- F1 Score: 0.406  
- ROC AUC: 0.675  

### Gradient Boosting + Feature Selection (Top 20 Features)
- Accuracy: 0.739  
- F1 Score: 0.523  
- ROC AUC: 0.727  

### Stacking Ensemble
- Accuracy: 0.723  
- F1 Score: 0.377  
- ROC AUC: 0.686  

## Running the Project

Install requirements:

```
pip install -r requirements.txt
```

Run experiments:

```
python -m src.phase2
```

## Possible Future Extensions
- SHAP or permutation feature importance
- Hyperparameter tuning
- Bayesian networks or probabilistic models
- Reinforcement learning for intervention allocation
- Deployment as an interactive dashboard or API

## Author
Ansh Patel  
GitHub: https://github.com/anshcpatel11
