#รัน01ก่อน
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"fraud": "#E74C3C", "normal": "#3498DB", "accent": "#2ECC71"}
os.makedirs("../assets", exist_ok=True)
os.makedirs("../models", exist_ok=True)

print("=" * 60)
print("โหลดข้อมูลที่ Preprocess แล้ว")
print("=" * 60)

data = joblib.load("../data/preprocessed_data.pkl")
X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]
feature_cols = data["feature_cols"]

print(f"✅ โหลดสำเร็จ: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test samples")


def evaluate_model(name, model, X_test, y_test, ax_roc, ax_pr, color):
    """ประเมินผลโมเดลและวาด ROC + PR Curve"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.4f})", color=color, lw=2)

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ax_pr.plot(rec, prec, label=f"{name} (AP={avg_prec:.4f})", color=color, lw=2)

    print(f"\n{'─'*50}")
    print(f"📊 {name}")
    print(f"{'─'*50}")
    print(f"  ROC-AUC:  {roc_auc:.4f}")
    print(f"  Avg Prec: {avg_prec:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal','Fraud'])}")

    return {"name": name, "roc_auc": roc_auc, "avg_prec": avg_prec,
            "f1": f1, "y_pred": y_pred, "y_prob": y_prob}


def plot_confusion_matrix(y_test, y_pred, title, ax):
    """วาด Confusion Matrix"""
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"],
                cbar=False)
    
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, f"({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=9, color="gray")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")


print("\n" + "=" * 60)
print("SECTION 3: SMOTE — Handling Class Imbalance")
print("=" * 60)

print("⚙️  ใช้ SMOTE (Synthetic Minority Over-sampling Technique)")
print("   เหตุผล: สร้าง synthetic fraud samples เฉพาะใน training set")
print("   ⚠️  ทำ SMOTE เฉพาะ training set เท่านั้น!")
print("       (ถ้าทำ SMOTE ก่อน split จะเกิด data leakage)")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"\n  Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"  After SMOTE:  {pd.Series(y_train_res).value_counts().to_dict()}")

print("\n" + "=" * 60)
print("SECTION 4: Training 3 Models (Bonus: Model Comparison)")
print("=" * 60)

fig_curves, ((ax_roc, ax_pr)) = plt.subplots(1, 2, figsize=(14, 5))
fig_curves.suptitle("ROC Curve & Precision-Recall Curve — Model Comparison",
                     fontsize=13, fontweight="bold")

results = {}
model_colors = ["#3498DB", "#2ECC71", "#E74C3C"]

print("\n🔵 Model 1: Logistic Regression (Baseline)")
print("   เหตุผลที่เลือก: เป็น baseline ที่ตีความผลได้ง่าย, เร็ว, เหมาะกับ imbalanced data")

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_train_res, y_train_res)
results["Logistic Regression"] = evaluate_model(
    "Logistic Regression", lr, X_test, y_test, ax_roc, ax_pr, model_colors[0])

print("\n🟢 Model 2: Random Forest")
print("   เหตุผลที่เลือก: Ensemble method ที่แข็งแกร่ง, ให้ feature importance ได้")
print("   ⚙️  Tuning hyperparameters ด้วย GridSearchCV...")

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
}
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
rf_cv = GridSearchCV(rf_base, rf_params, cv=StratifiedKFold(n_splits=3),
                     scoring="roc_auc", n_jobs=-1, verbose=0)
rf_cv.fit(X_train_res, y_train_res)

print(f"   ✅ Best params: {rf_cv.best_params_}")
print(f"   Best CV ROC-AUC: {rf_cv.best_score_:.4f}")

results["Random Forest"] = evaluate_model(
    "Random Forest", rf_cv.best_estimator_, X_test, y_test, ax_roc, ax_pr, model_colors[1])

print("\n🔴 Model 3: XGBoost (Gradient Boosting)")
print("   เหตุผลที่เลือก: State-of-the-art สำหรับ tabular data, จัดการ imbalance ได้ดี")
print("   ⚙️  Tuning hyperparameters ด้วย GridSearchCV...")

fraud_count = (y_train_res == 1).sum()
normal_count = (y_train_res == 0).sum()
scale_pos_weight = normal_count / fraud_count 

xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
}
xgb_base = xgb.XGBClassifier(
    random_state=42, scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr", use_label_encoder=False
)
xgb_cv = GridSearchCV(xgb_base, xgb_params, cv=StratifiedKFold(n_splits=3),
                       scoring="roc_auc", n_jobs=-1, verbose=0)
xgb_cv.fit(X_train_res, y_train_res)

print(f"   ✅ Best params: {xgb_cv.best_params_}")
print(f"   Best CV ROC-AUC: {xgb_cv.best_score_:.4f}")

results["XGBoost"] = evaluate_model(
    "XGBoost", xgb_cv.best_estimator_, X_test, y_test, ax_roc, ax_pr, model_colors[2])

ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend(loc="lower right")

ax_pr.axhline(y=y_test.mean(), color="k", linestyle="--", lw=1,
              label=f"Baseline ({y_test.mean():.4f})")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall Curve")
ax_pr.legend(loc="upper right")

plt.tight_layout()
plt.savefig("../assets/04_roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 บันทึกภาพ: ../assets/04_roc_pr_curves.png")

fig_cm, axes = plt.subplots(1, 3, figsize=(15, 4))
fig_cm.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")

for i, (name, res) in enumerate(results.items()):
    plot_confusion_matrix(y_test, res["y_pred"], name, axes[i])

plt.tight_layout()
plt.savefig("../assets/05_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 บันทึกภาพ: ../assets/05_confusion_matrices.png")

print("\n" + "=" * 60)
print("SECTION 6: Feature Importance (XGBoost)")
print("=" * 60)

best_model = xgb_cv.best_estimator_
feat_imp = pd.Series(
    best_model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("🔝 Top 10 important features:")
print(feat_imp.head(10).round(4))

fig, ax = plt.subplots(figsize=(10, 6))
feat_imp.head(15).plot(kind="barh", ax=ax, color=COLORS["fraud"], edgecolor="white")
ax.invert_yaxis()
ax.set_title("Top 15 Feature Importances (XGBoost)", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("../assets/06_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 บันทึกภาพ: ../assets/06_feature_importance.png")

print("\n" + "=" * 60)
print("SECTION 7: Model Comparison & Business Interpretation")
print("=" * 60)

summary = pd.DataFrame([
    {"Model": k, "ROC-AUC": v["roc_auc"],
     "Avg Precision": v["avg_prec"], "F1-Score": v["f1"]}
    for k, v in results.items()
]).set_index("Model")

print("\n📊 สรุปผลเปรียบเทียบโมเดล:")
print(summary.round(4).to_string())

xgb_res = results["XGBoost"]
y_pred_best = xgb_res["y_pred"]
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()

print(f"\n💼 Business Interpretation (XGBoost บน test set):")
print(f"   True Positive  (Fraud ถูกตรวจจับ):   {tp:4d} ธุรกรรม ✅")
print(f"   False Negative (Fraud ที่พลาด):       {fn:4d} ธุรกรรม ⚠️  ← อันตราย!")
print(f"   False Positive (Normal แต่ถูก flag):  {fp:4d} ธุรกรรม ℹ️  ← ทำให้ลูกค้าไม่พอใจ")
print(f"   True Negative  (Normal ถูกต้อง):      {tn:,} ธุรกรรม ✅")
print(f"\n   Recall = {tp/(tp+fn):.1%} → ตรวจพบ fraud จริง {tp/(tp+fn):.1%}")
print(f"   ใน context นี้ Recall สำคัญกว่า Precision เพราะ")
print(f"   การพลาด fraud มีต้นทุนสูงกว่าการ flag false alarm")

print("\n" + "=" * 60)
print("SECTION 8: บันทึก Best Model")
print("=" * 60)

model_package = {
    "model": xgb_cv.best_estimator_,
    "feature_cols": feature_cols,
    "scaler_amount": data["scaler_amount"],
    "scaler_time": data["scaler_time"],
    "best_params": xgb_cv.best_params_,
    "metrics": {
        "roc_auc": xgb_res["roc_auc"],
        "avg_precision": xgb_res["avg_prec"],
        "f1_score": xgb_res["f1"],
    },
    "class_names": ["Normal", "Fraud"],
}

joblib.dump(model_package, "../models/best_model_xgboost.pkl")
joblib.dump(rf_cv.best_estimator_, "../models/rf_model.pkl")
joblib.dump(lr, "../models/lr_model.pkl")

print("💾 บันทึกโมเดลสำเร็จ:")
print("   ../models/best_model_xgboost.pkl  (Best — ใช้ใน Streamlit App)")
print("   ../models/rf_model.pkl")
print("   ../models/lr_model.pkl")
print("\n🎉 Model Training & Evaluation เสร็จสมบูรณ์!")
print("   → รัน Streamlit app: cd ../app && streamlit run app.py")
