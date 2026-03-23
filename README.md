Credit Card Fraud Detection

ภาพรวมโปรเจค
โปรเจคนี้พัฒนาระบบ Machine Learning เพื่อตรวจจับธุรกรรมบัตรเครดิตที่ฉ้อโกง (Fraud Detection)
โดยใช้ข้อมูลธุรกรรมจริงที่ผ่านการทำ PCA anonymization แล้ว

ทำไมถึงเลือกปัญหานี้?
- Fraud มีต้นทุนสูงมากต่อทั้งธนาคารและผู้บริโภค
- มี class imbalance สูง (~0.17% เป็น fraud) → ต้องเลือก metrics อย่างระมัดระวัง
- เป็นปัญหาจริงที่ ML มีบทบาทสำคัญในอุตสาหกรรม FinTech

## Dataset
-ที่มา: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
-ข้อมูลขนาด: 284,807 transactions (492 fraud = 0.17%)
-Features:V1-V28 (PCA), Time, Amount, Class (target)

## โครงสร้างไฟล์
```
fraud_detection/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda_preprocessing.py   #EDA + Data Preprocessing รันก่อน
│   └── 02_model_evaluation.py    #Model Training + Evaluation รันหลัง
├── app/
│   └── app.py                    #Web App
├── models/                       #Save model
└── assets/                       #อันเก็บoutput plotเป็นรูปจาก 01_eda
```

วิธีรัน

1. ติดตั้ง dependencies
```bash
pip install -r requirements.txt
```

2. รัน EDA + Preprocessing
```bash
cd notebooks
python 01_eda_preprocessing.py
```

3. รัน Model Training
```bash
python 02_model_evaluation.py
```

4. รัน Streamlit App
```bash
cd app
streamlit run app.py
```

ผลลัพธ์หลัก
| Model | ROC-AUC | F1-Score (Fraud) | Precision | Recall |
|-------|---------|-----------------|-----------|--------|
| Logistic Regression | ~0.97 | ~0.72 | ~0.87 | ~0.61 |
| Random Forest | ~0.99 | ~0.86 | ~0.94 | ~0.79 |
| **XGBoost (Best)** | **~0.99** | **~0.88** | **~0.91** | **~0.84** |

Key Insights เสริมนิดหน่อย
- ไม่ควรใช้ Accuracy เป็น metric หลักเพราะ imbalanced dataset (99.83% non-fraud)
- ใช้ **SMOTE** แก้ปัญหา class imbalance ใน training set
- **Recall สำคัญกว่า Precision** ในบริบทนี้ (ดีกว่าจับ fraud ผิดพลาด มากกว่าพลาด fraud จริง)
