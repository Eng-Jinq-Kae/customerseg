import streamlit as st
import dataloader as dl
import pipeline as pipeline

st.title("📊 Customer Analytics Dashboard")
if 'file_upload' not in st.session_state or st.session_state.file_upload is None:
    st.warning("Go to page app to uplaod file.")
    st.stop()
else:
    st.markdown(":material/check_circle: Reading uplaoded file.")
    df = st.session_state.file_upload

# st.dataframe(df.head())
# pipeline.st_describe(df)
df = dl.read_df_preprocess(df)
df = pipeline.impute_missing_salary(df)

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Demographics",
    "Income Analysis",
    "Purchasing Analysis",
    "Membership Active Channel",
    "Recency",
    "Segmentation"
])

# =========================
# Demographics
# =========================
with tab1:
    st.subheader("Demographics")
    df = pipeline.add_customer_age(df)
    pipeline.chart_customer_age(df)
    st.divider()

# =========================
# Income Analysis
# =========================
with tab2:
    st.subheader("Income by Marital Status")
    pipeline.chart_customer_income_marital(df)
    st.subheader("Income Tier Distribution by Age (≤80)")
    df_use = df[df["Age"] <= 80]
    pipeline.chart_customer_income(df_use)
    st.subheader("Income Tier Distribution by Age (>80)")
    df_use = df[df["Age"] > 80]
    pipeline.chart_customer_income(df_use)
    st.divider()

# =========================
# Purchasing Analysis
# =========================
with tab3:
    st.subheader("Purchasing analysis")
    df_use = df[df["Age"] <= 80]
    pipeline.chart_customer_purchasing(df_use)
    df_use = df[df["Age"] > 80]
    pipeline.chart_customer_purchasing(df_use)
    st.divider()

# =========================
# Membership
# =========================
with tab4:
    st.subheader("Customer membership & activity")
    pipeline.chart_customer_membership(df)
    st.divider()

# =========================
# Recency
# =========================
with tab5:
    st.subheader("Customer recency")
    df_use = df[df["Age"] <= 80]
    pipeline.chart_customer_recency(df_use)
    df_use = df[df["Age"] > 80]
    pipeline.chart_customer_recency(df_use)
    st.divider()

# =========================
# Segmentation
# by Aqilah
# =========================
with tab6:
    st.subheader("Customer Preference Clustering")
    pipeline.chart_customer_segmentation(df)

st.success("Analysis page laoded full successfully.")