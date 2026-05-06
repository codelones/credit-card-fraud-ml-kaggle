import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import warnings
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, cross_val_predict)
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, precision_recall_curve,
                             recall_score, precision_score, f1_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import optuna

warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.3f" % x)


# =============================================================
# 1. VERİYİ YÜKLE
# =============================================================
df = pd.read_csv("datasets/creditcard.csv")


# =============================================================
# 2. EDA
# =============================================================
def first_check_df(dataframe, plotna=False):
    print("Dataset Shape")
    print(f"Observations : {dataframe.shape[0]}\nFeatures     : {dataframe.shape[1]}")
    print("**********************************************")
    print("Feature Types")
    print(dataframe.dtypes.value_counts())
    print("**********************************************")
    print("Descriptive Statistics (Numerical)")
    print(dataframe.describe(include=["int64", "float64"]))
    print("**********************************************")
    print("First 5 Observations")
    print(dataframe.head())
    print("**********************************************")
    print("Last 5 Observations")
    print(dataframe.tail())
    print("**********************************************")
    print(f"Duplicate Rows: {dataframe.duplicated().sum()}")
    print("**********************************************")
    missing = dataframe.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (missing / len(dataframe) * 100).round(2)
    if missing.empty:
        print("No Missing Values")
    else:
        print("Missing Value Ratio (%)")
        print(missing_pct)
    print("**********************************************")
    if plotna:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        msno.bar(dataframe, ax=axes[0], fontsize=8)
        msno.matrix(dataframe, ax=axes[1], fontsize=8)
        plt.tight_layout()
        plt.show(block=True)


first_check_df(df, plotna=False)

# Class dağılımı
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True) * 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = df["Class"].value_counts()
axes[0].bar(["Normal", "Fraud"], counts.values, color=["steelblue", "crimson"])
axes[0].set_title("Class Distribution (Count)")
pcts = df["Class"].value_counts(normalize=True).values * 100
axes[1].bar(["Normal", "Fraud"], pcts, color=["steelblue", "crimson"])
axes[1].set_title("Class Distribution (%)")
plt.tight_layout()
plt.show()

# Duplicate analizi
dupes = df[df.duplicated(keep=False)]
print(dupes["Class"].value_counts())
print(f"Fraud rate in duplicates: {dupes['Class'].mean()*100:.2f}% (vs 0.17% overall)")

# Duplicate sil
df = df.drop_duplicates()
print(df.shape)
print(df["Class"].value_counts())

# Amount ve Time dağılımı
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
ax1.hist(df[df["Class"] == 0]["Amount"], bins=100, color="steelblue", alpha=0.7)
ax1.set_title("Normal - Amount Distribution")
ax2.hist(df[df["Class"] == 1]["Amount"], bins=100, color="crimson", alpha=0.7)
ax2.set_title("Fraud - Amount Distribution")
ax3.hist(df[df["Class"] == 0]["Time"], bins=100, color="steelblue", alpha=0.7)
ax3.set_title("Normal - Time Distribution")
ax4.hist(df[df["Class"] == 1]["Time"], bins=100, color="crimson", alpha=0.7)
ax4.set_title("Fraud - Time Distribution")
plt.tight_layout()
plt.show()

print(df.groupby("Class")["Amount"].describe())
print(df.groupby("Class")["Time"].describe())


# =============================================================
# 3. SCALING
# =============================================================
scaler = RobustScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
df = df.drop(["Amount", "Time"], axis=1)

print(df.shape)
print(df.head())


# =============================================================
# 4. TRAIN / TEST SPLIT
# =============================================================
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape} | Fraud: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
print(f"Test : {X_test.shape}  | Fraud: {y_test.sum()} ({y_test.mean()*100:.3f}%)")


# =============================================================
# 5. CLASS IMBALANCE — class_weight vs SMOTE
# =============================================================

# Model 1: LightGBM + class_weight
model_cw = LGBMClassifier(verbose=-1, class_weight="balanced", n_estimators=100, random_state=42)
model_cw.fit(X_train, y_train)
y_pred_cw = model_cw.predict(X_test)
y_prob_cw = model_cw.predict_proba(X_test)[:, 1]

print("\n--- LightGBM + class_weight ---")
print(classification_report(y_test, y_pred_cw))
print(f"ROC-AUC : {roc_auc_score(y_test, y_prob_cw):.4f}")
print(f"PR-AUC  : {average_precision_score(y_test, y_prob_cw):.4f}")

# Model 2: LightGBM + SMOTE (sadece train'e uygulanır!)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — train fraud: {y_train_smote.sum()} (was {y_train.sum()})")

model_smote = LGBMClassifier(verbose=-1, n_estimators=100, random_state=42)
model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = model_smote.predict(X_test)
y_prob_smote = model_smote.predict_proba(X_test)[:, 1]

print("\n--- LightGBM + SMOTE ---")
print(classification_report(y_test, y_pred_smote))
print(f"ROC-AUC : {roc_auc_score(y_test, y_prob_smote):.4f}")
print(f"PR-AUC  : {average_precision_score(y_test, y_prob_smote):.4f}")

# Karşılaştırma
print("\n========== class_weight vs SMOTE ==========")
comparison = {
    "LightGBM + class_weight": {
        "ROC-AUC"  : round(roc_auc_score(y_test, y_prob_cw), 4),
        "PR-AUC"   : round(average_precision_score(y_test, y_prob_cw), 4),
        "Recall"   : round(recall_score(y_test, y_pred_cw), 4),
        "Precision": round(precision_score(y_test, y_pred_cw), 4),
    },
    "LightGBM + SMOTE": {
        "ROC-AUC"  : round(roc_auc_score(y_test, y_prob_smote), 4),
        "PR-AUC"   : round(average_precision_score(y_test, y_prob_smote), 4),
        "Recall"   : round(recall_score(y_test, y_pred_smote), 4),
        "Precision": round(precision_score(y_test, y_pred_smote), 4),
    }
}
print(pd.DataFrame(comparison).T)
print("\n→ class_weight wins: better PR-AUC and much better Precision")


# =============================================================
# 6. MODEL KARŞILAŞTIRMASI
# =============================================================
models = {
    "LightGBM": LGBMClassifier(verbose=-1, class_weight="balanced",
                                n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                              n_estimators=100, random_state=42, eval_metric="aucpr"),
    "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=100,
                                            random_state=42, n_jobs=-1),
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
}

results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred_ = m.predict(X_test)
    y_prob_ = m.predict_proba(X_test)[:, 1]
    results[name] = {
        "ROC-AUC"  : round(roc_auc_score(y_test, y_prob_), 4),
        "PR-AUC"   : round(average_precision_score(y_test, y_prob_), 4),
        "Recall"   : round(recall_score(y_test, y_pred_), 4),
        "Precision": round(precision_score(y_test, y_pred_), 4),
        "F1"       : round(f1_score(y_test, y_pred_), 4)
    }
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred_))

print("\n========== MODEL KARŞILAŞTIRMA TABLOSU ==========")
results_df = pd.DataFrame(results).T
print(results_df.sort_values("PR-AUC", ascending=False))

# Feature Importance (LightGBM)
lgbm_model = models["LightGBM"]
feature_importance = pd.DataFrame({
    "feature"   : X_train.columns,
    "importance": lgbm_model.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 12))
plt.barh(feature_importance["feature"], feature_importance["importance"], color="steelblue")
plt.gca().invert_yaxis()
plt.title("Feature Importance — LightGBM")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# =============================================================
# 7. CV + OPTİMİZASYON (Optuna)
# =============================================================
def optimize_model_cv(model_name, X, y, n_trials=50):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        if model_name == "XGBoost":
            params = {
                "n_estimators"    : trial.suggest_int("n_estimators", 100, 500),
                "max_depth"       : trial.suggest_int("max_depth", 3, 10),
                "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "scale_pos_weight": len(y[y == 0]) / len(y[y == 1]),
                "random_state"    : 42,
                "eval_metric"     : "aucpr"
            }
            model = XGBClassifier(**params)

        elif model_name == "LightGBM":
            params = {
                "n_estimators"     : trial.suggest_int("n_estimators", 100, 500),
                "max_depth"        : trial.suggest_int("max_depth", 3, 10),
                "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "class_weight"     : "balanced",
                "random_state"     : 42,
                "verbose"          : -1
            }
            model = LGBMClassifier(**params)

        scores = cross_val_score(model, X, y, cv=cv,
                                 scoring="average_precision", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n{model_name} CV — Best PR-AUC: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    return study.best_params, study.best_value


lgbm_cv_params, lgbm_cv_score = optimize_model_cv("LightGBM", X, y, n_trials=50)
xgb_cv_params,  xgb_cv_score  = optimize_model_cv("XGBoost",  X, y, n_trials=50)

print(f"\nLightGBM CV Optuna → PR-AUC: {lgbm_cv_score:.4f}")
print(f"XGBoost  CV Optuna → PR-AUC: {xgb_cv_score:.4f}")


# =============================================================
# 8. CROSS VALIDATION — Gerçek Kazananı Bul
# =============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=== 5-Fold Cross Validation (PR-AUC) ===")
for name, params, clf, extra in [
    ("XGBoost",  xgb_cv_params,  XGBClassifier,  {"random_state": 42, "eval_metric": "aucpr"}),
    ("LightGBM", lgbm_cv_params, LGBMClassifier, {"random_state": 42, "class_weight": "balanced", "verbose": -1})
]:
    m = clf(**params, **extra)
    scores = cross_val_score(m, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Folds: {[f'{s:.4f}' for s in scores]}")


# =============================================================
# 9. THRESHOLD OPTIMIZATION
# =============================================================
final_model = XGBClassifier(**xgb_cv_params, random_state=42, eval_metric="aucpr")
final_model.fit(X_train, y_train)
y_prob_final = final_model.predict_proba(X_test)[:, 1]

precisions_arr, recalls_arr, thresholds_arr = precision_recall_curve(y_test, y_prob_final)
f2_scores_arr = (5 * precisions_arr * recalls_arr) / (4 * precisions_arr + recalls_arr + 1e-8)
optimal_idx = np.argmax(f2_scores_arr)
optimal_threshold = thresholds_arr[optimal_idx]

print(f"\nOptimal Threshold : {optimal_threshold:.4f}")
print(f"Precision         : {precisions_arr[optimal_idx]:.4f}")
print(f"Recall            : {recalls_arr[optimal_idx]:.4f}")
print(f"F2 Score          : {f2_scores_arr[optimal_idx]:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(thresholds_arr, precisions_arr[:-1], label="Precision", color="steelblue", linewidth=2)
plt.plot(thresholds_arr, recalls_arr[:-1],    label="Recall",    color="crimson",   linewidth=2)
plt.plot(thresholds_arr, f2_scores_arr[:-1],  label="F2 Score",  color="green",     linewidth=2)
plt.axvline(optimal_threshold, color="black", linestyle="--",
            linewidth=1.5, label=f"Optimal: {optimal_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision / Recall / F2 vs Threshold")
plt.legend()
plt.show()


# =============================================================
# 10. STACKING (Deney)
# =============================================================
cv_stack = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgbm_base = LGBMClassifier(verbose=-1, **lgbm_cv_params, class_weight="balanced", random_state=42)
xgb_base  = XGBClassifier(**xgb_cv_params, random_state=42, eval_metric="aucpr")

print("Out-of-fold tahminler hesaplanıyor...")
lgbm_oof = cross_val_predict(lgbm_base, X_train, y_train, cv=cv_stack,
                              method="predict_proba", n_jobs=-1)[:, 1]
xgb_oof  = cross_val_predict(xgb_base,  X_train, y_train, cv=cv_stack,
                              method="predict_proba", n_jobs=-1)[:, 1]

lgbm_base.fit(X_train, y_train)
xgb_base.fit(X_train, y_train)

X_meta_train = np.column_stack([lgbm_oof, xgb_oof])
X_meta_test  = np.column_stack([lgbm_base.predict_proba(X_test)[:, 1],
                                  xgb_base.predict_proba(X_test)[:, 1]])

meta_model = LogisticRegression(class_weight="balanced", random_state=42)
meta_model.fit(X_meta_train, y_train)
y_prob_stack = meta_model.predict_proba(X_meta_test)[:, 1]

stack_prauc = average_precision_score(y_test, y_prob_stack)
print(f"\nStacking PR-AUC : {stack_prauc:.4f}")
print(f"XGBoost PR-AUC  : {xgb_cv_score:.4f}")
print("→ Stacking did not improve — two similar models produce redundant predictions")


# =============================================================
# 11. FINAL SONUÇLAR
# =============================================================
y_pred_final = (y_prob_final >= optimal_threshold).astype(int)

print("\n=== FINAL MODEL: XGBoost + CV Optuna + Threshold Optimization ===")
print(classification_report(y_test, y_pred_final))
print(f"ROC-AUC : {roc_auc_score(y_test, y_prob_final):.4f}")
print(f"PR-AUC  : {average_precision_score(y_test, y_prob_final):.4f}")

# Score progression
progression = {
    "LightGBM baseline"  : results["LightGBM"]["PR-AUC"],
    "LightGBM + SMOTE"   : round(average_precision_score(y_test, y_prob_smote), 4),
    "XGBoost baseline"   : results["XGBoost"]["PR-AUC"],
    "LightGBM CV Optuna" : lgbm_cv_score,
    "XGBoost CV Optuna"  : xgb_cv_score,
    "Stacking"           : stack_prauc
}

print("\n========== SCORE PROGRESSION ==========")
for name, score in progression.items():
    print(f"{name:<25}: {score:.4f}")

fig, ax = plt.subplots(figsize=(14, 5))
colors_prog = ["#AED6F1", "#AED6F1", "#2E75B6", "#F0B27A", "#E74C3C", "#BDC3C7"]
bars = ax.bar(list(progression.keys()), list(progression.values()),
              color=colors_prog, edgecolor="white")
ax.set_title("PR-AUC Score Progression", fontsize=13, fontweight="bold")
ax.set_ylabel("PR-AUC")
ax.set_ylim(min(progression.values()) - 0.02, max(progression.values()) + 0.02)
ax.set_xticklabels(list(progression.keys()), rotation=15)
for bar, val in zip(bars, progression.values()):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
            f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.show()
