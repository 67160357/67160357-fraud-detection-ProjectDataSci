"""
=============================================================
01_eda_preprocessing.py
Credit Card Fraud Detection — EDA & Preprocessing
=============================================================
Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
ดาวน์โหลด creditcard.csv แล้ววางไว้ใน ../data/creditcard.csv
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"fraud": "#E74C3C", "normal": "#3498DB", "accent": "#2ECC71"}

os.makedirs("../assets", exist_ok=True)
os.makedirs("../data", exist_ok=True)

print("=" * 60)
print("SECTION 1: โหลดข้อมูล")
print("=" * 60)

df = pd.read_csv("../data/creditcard.csv")

print(f"✅ โหลดข้อมูลสำเร็จ")
print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n📊 ประเภทข้อมูล:\n{df.dtypes.value_counts()}")
print(f"\n📋 5 แถวแรก:\n{df.head()}")

print("\n" + "=" * 60)
print("SECTION 2: ตรวจสอบ Missing Values")
print("=" * 60)

missing = df.isnull().sum()
print(f"Missing values รวม: {missing.sum()}")
if missing.sum() == 0:
    print("✅ ไม่มี missing values ในชุดข้อมูลนี้")
    print("   เนื่องจาก dataset ถูก preprocess มาจาก Kaggle แล้ว")
else:
    print(f"⚠️ พบ missing values:\n{missing[missing > 0]}")

dups = df.duplicated().sum()
print(f"\nDuplicate rows: {dups:,}")
if dups > 0:
    df = df.drop_duplicates()
    print(f"✅ ลบ duplicates แล้ว → เหลือ {len(df):,} rows")

print("\n" + "=" * 60)
print("SECTION 3: Class Distribution (Class Imbalance)")
print("=" * 60)

class_counts = df["Class"].value_counts()
fraud_pct = class_counts[1] / len(df) * 100

print(f"Normal (0): {class_counts[0]:,}  ({100 - fraud_pct:.2f}%)")
print(f"Fraud  (1): {class_counts[1]:,}  ({fraud_pct:.4f}%)")
print(f"\n⚠️  IMBALANCE RATIO = 1 : {class_counts[0]//class_counts[1]}")
print("   → ถ้าโมเดลทำนาย 0 ทุกครั้ง จะได้ Accuracy = 99.83%")
print("   → นั่นหมายความว่า Accuracy ไม่ใช่ metrics ที่เหมาะสม")
print("   → ควรใช้ Precision, Recall, F1, ROC-AUC แทน")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Class Distribution — Credit Card Fraud Dataset", fontsize=14, fontweight="bold")

axes[0].bar(["Normal", "Fraud"], class_counts.values,
            color=[COLORS["normal"], COLORS["fraud"]], edgecolor="white", linewidth=1.5)
axes[0].set_title("Transaction Count")
axes[0].set_ylabel("Count")
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")

axes[1].pie([class_counts[0], class_counts[1]],
            labels=[f"Normal\n({100-fraud_pct:.2f}%)", f"Fraud\n({fraud_pct:.4f}%)"],
            colors=[COLORS["normal"], COLORS["fraud"]],
            autopct="%1.2f%%", startangle=90, explode=(0, 0.1))
axes[1].set_title("Class Proportion")

plt.tight_layout()
plt.savefig("../assets/01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 บันทึกภาพ: ../assets/01_class_distribution.png")

print("\n" + "=" * 60)
print("SECTION 4: Descriptive Statistics")
print("=" * 60)

print("\n📊 Amount statistics by class:")
amount_stats = df.groupby("Class")["Amount"].describe()
amount_stats.index = ["Normal", "Fraud"]
print(amount_stats.round(2))

print("\n📊 Time statistics:")
print(f"   ช่วงเวลา: {df['Time'].min()/3600:.1f} - {df['Time'].max()/3600:.1f} ชั่วโมง")

print("\n" + "=" * 60)
print("SECTION 5: EDA — Amount Analysis")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Transaction Amount Analysis", fontsize=14, fontweight="bold")

fraud_df = df[df["Class"] == 1]
normal_df = df[df["Class"] == 0]

axes[0].hist(normal_df["Amount"].clip(upper=1000), bins=80,
             alpha=0.7, color=COLORS["normal"], label="Normal", density=True)
axes[0].hist(fraud_df["Amount"].clip(upper=1000), bins=80,
             alpha=0.7, color=COLORS["fraud"], label="Fraud", density=True)
axes[0].set_xlabel("Amount (clipped at 1000)")
axes[0].set_ylabel("Density")
axes[0].set_title("Amount Distribution (Normal vs Fraud)")
axes[0].legend()

data_to_plot = [normal_df["Amount"].clip(upper=500), fraud_df["Amount"].clip(upper=500)]
bp = axes[1].boxplot(data_to_plot, labels=["Normal", "Fraud"],
                     patch_artist=True,
                     boxprops=dict(facecolor="lightblue"),
                     medianprops=dict(color="red", linewidth=2))
bp["boxes"][1].set_facecolor("lightsalmon")
axes[1].set_title("Amount Boxplot (clipped at 500)")
axes[1].set_ylabel("Amount")

plt.tight_layout()
plt.savefig("../assets/02_amount_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 บันทึกภาพ: ../assets/02_amount_analysis.png")

print("\n" + "=" * 60)
print("SECTION 6: Correlation with Fraud")
print("=" * 60)

corr_with_fraud = df.corr()["Class"].drop("Class").sort_values()
top_positive = corr_with_fraud.tail(5)
top_negative = corr_with_fraud.head(5)

print("🔴 Features ที่สัมพันธ์กับ Fraud มากที่สุด (positive):")
print(top_positive.round(4))
print("\n🔵 Features ที่สัมพันธ์กับ Non-Fraud มากที่สุด (negative):")
print(top_negative.round(4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Feature Correlation with Fraud", fontsize=14, fontweight="bold")

all_corr = pd.concat([top_negative, top_positive])
colors = [COLORS["normal"] if x < 0 else COLORS["fraud"] for x in all_corr.values]
axes[0].barh(all_corr.index, all_corr.values, color=colors, edgecolor="white")
axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_title("Top 10 Features Correlated with Fraud")
axes[0].set_xlabel("Correlation Coefficient")

top_feats = list(top_negative.index[:2]) + list(top_positive.index[-2:])
for i, feat in enumerate(top_feats[:4]):
    ax = axes[1] if i == 0 else None

axes[1].hist(normal_df["V14"], bins=80, alpha=0.6,
             color=COLORS["normal"], label="Normal", density=True)
axes[1].hist(fraud_df["V14"], bins=80, alpha=0.6,
             color=COLORS["fraud"], label="Fraud", density=True)
axes[1].set_title("V14 Distribution (Strongest Negative Corr.)")
axes[1].set_xlabel("V14 Value")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.savefig("../assets/03_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 บันทึกภาพ: ../assets/03_correlation.png")

print("\n" + "=" * 60)
print("SECTION 7: Preprocessing")
print("=" * 60)

print("🔧 Scaling Amount และ Time:")
print("   - V1-V28: ถูก scale มาแล้วจาก PCA")
print("   - Amount: ต้องทำ log transform + StandardScaler (skewed มาก)")
print("   - Time: ทำ StandardScaler")

df["Amount_scaled"] = np.log1p(df["Amount"])  
scaler_amount = StandardScaler()
scaler_time = StandardScaler()

df["Amount_scaled"] = scaler_amount.fit_transform(df[["Amount"]])
df["Time_scaled"] = scaler_time.fit_transform(df[["Time"]])

feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
X = df[feature_cols]
y = df["Class"]

print(f"\n✅ Features ที่ใช้: {len(feature_cols)} features")
print(f"   V1-V28 (PCA features) + Amount_scaled + Time_scaled")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Train/Test Split (stratified):")
print(f"   Training set: {len(X_train):,} samples")
print(f"   Test set:     {len(X_test):,} samples")
print(f"   Fraud in train: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
print(f"   Fraud in test:  {y_test.sum():,} ({y_test.mean()*100:.3f}%)")

import joblib

joblib.dump({
    "X_train": X_train, "X_test": X_test,
    "y_train": y_train, "y_test": y_test,
    "feature_cols": feature_cols,
    "scaler_amount": scaler_amount,
    "scaler_time": scaler_time
}, "../data/preprocessed_data.pkl")

print("\n💾 บันทึก preprocessed data: ../data/preprocessed_data.pkl")
print("\n🎉 EDA & Preprocessing เสร็จสมบูรณ์!")
print("   → รัน 02_model_evaluation.py ต่อได้เลย")
