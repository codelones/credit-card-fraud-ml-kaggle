# 💳 Credit Card Fraud Detection

> **CV PR-AUC: 0.8543** | XGBoost + 5-Fold CV Optuna + Threshold Optimization

---

## 📋 Project Overview

A complete fraud detection pipeline on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The core challenge is extreme **class imbalance** — only 0.17% of transactions are fraud.
This project explores multiple techniques to handle imbalance, optimize models, and tune the decision threshold.

---

## 🗺️ Pipeline

```
Raw Data → EDA → Duplicate Removal → RobustScaler → Stratified Split
→ class_weight vs SMOTE → Multi-model comparison
→ CV + Optuna optimization → Threshold optimization (F2)
→ Cross-validation comparison → Stacking experiment → Final model
```

---

## 📊 Results

| Approach | PR-AUC | Recall | Precision |
|----------|--------|--------|-----------|
| LightGBM + class_weight (baseline) | 0.8257 | 0.79 | 0.82 |
| LightGBM + SMOTE | 0.8140 | 0.81 | 0.57 |
| XGBoost baseline | 0.8237 | 0.77 | 0.94 |
| LightGBM + CV Optuna | 0.8534 | — | — |
| Stacking (XGB + LGBM) | 0.8237 | — | — |
| **XGBoost + CV Optuna + Threshold** | **0.8543** | **0.80** | **0.92** |

---

## 💡 Key Insights

1. **Accuracy is misleading** — 99.83% accuracy = zero fraud caught. Always use PR-AUC for imbalanced data
2. **PR-AUC > ROC-AUC** — ROC-AUC is inflated by the massive true-negative pool
3. **class_weight > SMOTE** — SMOTE hurt precision badly (0.82 → 0.57) for minimal recall gain
4. **Single split is unreliable** — CV Optuna revealed XGBoost as the true winner
5. **Threshold optimization matters** — F2-based tuning balances asymmetric misclassification costs
6. **Stacking didn't help** — two similar gradient boosting models produce redundant predictions

---

## 🛠️ Techniques Used

- **Class Imbalance**: class_weight vs SMOTE comparison
- **Models**: LightGBM, XGBoost, Random Forest, Logistic Regression
- **Optimization**: Optuna with 5-Fold Cross Validation inside each trial
- **Threshold Tuning**: F2 score optimization (Recall weighted 2x over Precision)
- **Ensemble**: Stacking with OOF predictions + Logistic Regression meta-model

---

## 📁 Structure

```
credit-card-fraud/
│
├── datasets/
│   └── creditcard.csv          ← not included (download from Kaggle)
│
├── credit_card_fraud.py        ← main script
└── README.md
```

---

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib scikit-learn lightgbm xgboost imbalanced-learn optuna

# Download dataset from Kaggle
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in datasets/ folder

# Run
python credit_card_fraud.py
```

---

## 📚 What I Learned

| Concept | Insight |
|---------|---------|
| Accuracy paradox | 99.83% accuracy = catch zero fraud |
| PR-AUC vs ROC-AUC | PR-AUC is the honest metric for imbalanced data |
| class_weight vs SMOTE | class_weight won — less noise, better precision |
| CV importance | Single split showed wrong winner — always use CV |
| Threshold optimization | Default 0.50 is rarely optimal for fraud |
| Stacking limits | Needs diverse base models to be effective |

---

## 🔗 Links

- **Kaggle Notebook**: [Credit Card Fraud Detection](https://www.kaggle.com/hasanpireci)
- **Dataset**: [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

**Author**: Hasan Pireci | [GitHub](https://github.com/codelones) | hasanpireci92@gmail.com
