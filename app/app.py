"""
app.py — Credit Card Fraud Detection Web Application
รัน: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .fraud-alert {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        animation: pulse 2s infinite;
    }
    .safe-alert {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .disclaimer {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #856404;
        margin-top: 1rem;
    }
    .feature-help {
        font-size: 0.75rem;
        color: #6c757d;
        font-style: italic;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "../models/best_model_xgboost.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model_pkg = load_model()

# ─── Header ──────────────────────────────────────────────────
st.markdown('<div class="main-header">🛡️ Credit Card Fraud Detector</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">ระบบตรวจจับธุรกรรมบัตรเครดิตที่ผิดปกติด้วย Machine Learning (XGBoost)</div>',
            unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# Sidebar — Model Info
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 ข้อมูลโมเดล")
    st.markdown("**Algorithm:** XGBoost")
    st.markdown("**Dataset:** 284,807 transactions")

    if model_pkg:
        metrics = model_pkg.get("metrics", {})
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        st.metric("F1-Score (Fraud)", f"{metrics.get('f1_score', 0):.4f}")
        st.metric("Avg Precision", f"{metrics.get('avg_precision', 0):.4f}")
        st.markdown("---")
        params = model_pkg.get("best_params", {})
        st.markdown("**Best Hyperparameters:**")
        for k, v in params.items():
            st.markdown(f"  - `{k}`: {v}")
    else:
        st.warning("⚠️ โหลดโมเดลไม่สำเร็จ\nรัน 02_model_evaluation.py ก่อน")

    st.markdown("---")
    st.markdown("### ℹ️ เกี่ยวกับ Features")
    st.markdown("""
**V1-V28:** ค่าที่ได้จาก PCA (Principal Component Analysis)
เป็นการแปลงข้อมูลเพื่อปกป้องความเป็นส่วนตัวของลูกค้า

**Amount:** ยอดธุรกรรม (USD)

**Time:** เวลาตั้งแต่ธุรกรรมแรก (วินาที)
    """)

# ════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍 ทำนายธุรกรรม", "📈 Feature Importance", "ℹ️ วิธีการใช้งาน"])

# ─── TAB 1: Prediction ───────────────────────────────────────
with tab1:
    st.markdown("### กรอกข้อมูลธุรกรรม")
    st.markdown("กรอกค่า features ของธุรกรรมที่ต้องการตรวจสอบ:")

    col_info, col_form = st.columns([1, 2])

    with col_info:
        st.info("""
**💡 วิธีการป้อนข้อมูล**

1. ป้อนค่า V1–V28 ที่ได้จากระบบ
2. ป้อน Amount (ยอดเงิน)
3. ป้อน Time (วินาทีนับจากธุรกรรมแรก)
4. กด **ทำนาย** เพื่อดูผล

**ค่าตัวอย่าง:**
- ธุรกรรมปกติ: V1 ≈ -0.5 ถึง 2.0
- ธุรกรรมต้องสงสัย: V14 < -5 (signal หลัก)
        """)

    with col_form:
        # ─── Input form ─────────────────────────────────────
        with st.expander("📝 V1 – V14", expanded=True):
            cols = st.columns(4)
            v_values = {}
            for i in range(1, 15):
                col_idx = (i - 1) % 4
                v_values[f"V{i}"] = cols[col_idx].number_input(
                    f"V{i}", value=0.0, format="%.4f",
                    help=f"PCA component {i} (ปกติ: -3 ถึง 3)"
                )

        with st.expander("📝 V15 – V28"):
            cols = st.columns(4)
            for i in range(15, 29):
                col_idx = (i - 15) % 4
                v_values[f"V{i}"] = cols[col_idx].number_input(
                    f"V{i}", value=0.0, format="%.4f",
                    help=f"PCA component {i} (ปกติ: -3 ถึง 3)"
                )

        col_a, col_t = st.columns(2)
        with col_a:
            amount = col_a.number_input(
                "💰 Amount (USD)", min_value=0.0, max_value=25000.0,
                value=100.0, step=0.01,
                help="ยอดเงินของธุรกรรม (0 - 25,000 USD)"
            )
        with col_t:
            time_val = col_t.number_input(
                "⏱️ Time (วินาที)", min_value=0, max_value=172800,
                value=50000,
                help="เวลาตั้งแต่ธุรกรรมแรก (0 = เริ่มต้น, 172800 = 48 ชั่วโมง)"
            )

        # ─── Validation ──────────────────────────────────────
        if amount < 0:
            st.error("❌ Amount ต้องไม่ติดลบ")
        elif amount > 25000:
            st.warning("⚠️ Amount สูงผิดปกติ (>25,000 USD) — ตรวจสอบค่าอีกครั้ง")

        predict_btn = st.button("🔍 ทำนายธุรกรรม", type="primary", use_container_width=True)

    # ─── Prediction Result ───────────────────────────────────
    if predict_btn:
        if model_pkg is None:
            st.error("❌ ไม่พบโมเดล — กรุณารัน 02_model_evaluation.py ก่อน")
        else:
            model = model_pkg["model"]
            scaler_amount = model_pkg["scaler_amount"]
            scaler_time = model_pkg["scaler_time"]
            feature_cols = model_pkg["feature_cols"]

            # Scale inputs
            amount_scaled = scaler_amount.transform([[amount]])[0][0]
            time_scaled = scaler_time.transform([[time_val]])[0][0]

            # Build feature vector
            feature_vector = np.array(
                [v_values[f"V{i}"] for i in range(1, 29)] +
                [amount_scaled, time_scaled]
            ).reshape(1, -1)

            # Predict
            pred = model.predict(feature_vector)[0]
            prob = model.predict_proba(feature_vector)[0]
            fraud_prob = prob[1]
            normal_prob = prob[0]

            st.markdown("---")
            st.markdown("### 🎯 ผลการทำนาย")

            result_col, gauge_col = st.columns([1, 1])

            with result_col:
                if pred == 1:
                    st.markdown(f"""
<div class="fraud-alert">
⚠️ ตรวจพบธุรกรรมต้องสงสัย!<br>
<span style="font-size:0.9rem; font-weight:400;">
ความน่าจะเป็นว่าเป็น Fraud: <strong>{fraud_prob*100:.1f}%</strong>
</span>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div class="safe-alert">
✅ ธุรกรรมปกติ<br>
<span style="font-size:0.9rem; font-weight:400;">
ความน่าจะเป็นว่าปกติ: <strong>{normal_prob*100:.1f}%</strong>
</span>
</div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                m1.metric("Fraud Probability", f"{fraud_prob*100:.2f}%",
                          delta="⚠️ High" if fraud_prob > 0.5 else "✅ Low")
                m2.metric("Normal Probability", f"{normal_prob*100:.2f}%")

            with gauge_col:
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fraud_prob * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Fraud Risk Score", "font": {"size": 16}},
                    number={"suffix": "%", "font": {"size": 28}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": "#E74C3C" if fraud_prob > 0.5 else "#2ECC71"},
                        "steps": [
                            {"range": [0, 30], "color": "#d4edda"},
                            {"range": [30, 70], "color": "#fff3cd"},
                            {"range": [70, 100], "color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(t=30, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Disclaimer
            st.markdown("""
<div class="disclaimer">
⚠️ <strong>Disclaimer:</strong> ผลการทำนายนี้เป็นเพียงการประมาณการจากโมเดล ML เท่านั้น
ไม่ใช่การวินิจฉัยขั้นสุดท้าย การตัดสินใจเกี่ยวกับธุรกรรมควรได้รับการตรวจสอบจากผู้เชี่ยวชาญ
และระบบรักษาความปลอดภัยของสถาบันการเงินเสมอ
</div>""", unsafe_allow_html=True)

# ─── TAB 2: Feature Importance ───────────────────────────────
with tab2:
    st.markdown("### 📊 Feature Importance (XGBoost)")
    st.markdown("แสดง features ที่โมเดลให้ความสำคัญมากที่สุดในการตัดสินใจ")

    if model_pkg:
        model = model_pkg["model"]
        feature_cols = model_pkg["feature_cols"]
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=True).tail(15)

        fig_imp = px.bar(
            x=importances.values,
            y=importances.index,
            orientation="h",
            color=importances.values,
            color_continuous_scale="Reds",
            labels={"x": "Importance Score", "y": "Feature"},
            title="Top 15 Feature Importances"
        )
        fig_imp.update_layout(
            height=500,
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"}
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.info("""
**การตีความ:** Features ที่มี importance สูงสุด (เช่น V14, V4, V12)
คือ PCA components ที่แยกแยะ fraud กับ non-fraud ได้ดีที่สุด
แม้เราจะไม่ทราบความหมายของ V ในบริบทจริงเนื่องจากถูก anonymize แล้ว
แต่โมเดลสามารถใช้ pattern เหล่านี้ในการตัดสินใจได้
        """)
    else:
        st.warning("⚠️ ต้องรันโมเดลก่อนเพื่อดู Feature Importance")

# ─── TAB 3: How to Use ────────────────────────────────────────
with tab3:
    st.markdown("### ℹ️ วิธีการใช้งานและคำอธิบาย")
    st.markdown("""
#### 🎯 วัตถุประสงค์
ระบบนี้ใช้ Machine Learning (XGBoost) เพื่อตรวจจับธุรกรรมบัตรเครดิตที่อาจเป็นการฉ้อโกง
โดยวิเคราะห์จาก features ของธุรกรรมแต่ละรายการ

#### 📥 ข้อมูลที่ต้องป้อน

| Feature | คำอธิบาย | ค่าปกติ |
|---------|----------|---------|
| V1–V28 | ค่าที่ได้จาก PCA (anonymized) | ประมาณ -3 ถึง 3 |
| Amount | ยอดเงินธุรกรรม (USD) | 0 – 25,000 |
| Time | เวลาตั้งแต่ธุรกรรมแรก (วินาที) | 0 – 172,800 |

#### 📤 ผลลัพธ์
- **ผลการทำนาย:** Normal หรือ Fraud
- **Fraud Probability:** ความน่าจะเป็นที่จะเป็น fraud (0–100%)
- **Risk Gauge:** มาตรวัดความเสี่ยง (เขียว/เหลือง/แดง)

#### ⚠️ ข้อจำกัด
- โมเดลถูก train จากข้อมูลในยุโรปปี 2013
- Class imbalance: fraud เพียง 0.17% ของข้อมูลทั้งหมด
- ผลลัพธ์เป็นเพียงการประมาณการ ไม่ใช่การวินิจฉัยขั้นสุดท้าย
    """)