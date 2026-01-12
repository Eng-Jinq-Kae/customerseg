import streamlit as st
import dataloader as dl
import pipeline as pipeline
import pandas as pd

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
    st.write("As customer age >80 is considered a small group data, they should be observed seperately")
    st.divider()

# =========================
# Income Analysis
# =========================
with tab2:
    st.subheader("Income by Marital Status")
    pipeline.chart_customer_income_marital(df)
    st.write("Target those age has more T1 proportion")
    with st.expander("By Age (≤80)"):
        st.subheader("Income Tier Distribution by Age (≤80)")
        df_use = df[df["Age"] <= 80]
        pipeline.chart_customer_income(df_use)
    with st.expander("By Age (>80)"):
        st.subheader("Income Tier Distribution by Age (>80)")
        df_use = df[df["Age"] > 80]
        pipeline.chart_customer_income(df_use)
    st.divider()

# =========================
# Purchasing Analysis
# =========================
with tab3:
    st.write("In different products, target the age group has more spending, understand why.")
    st.subheader("Purchasing analysis")
    df_use_less_80 = df[df["Age"] <= 80]
    df_spend_less_80 = pipeline.customer_spend_category(df_use_less_80)
    df_spend_less_80 = df_spend_less_80.rename(columns={"Average_Spend":"Average_Spend_≤80 ($)"})
    df_use_more_80 = df[df["Age"] > 80]
    df_spend_more_80 = pipeline.customer_spend_category(df_use_more_80)
    df_spend_more_80 = df_spend_more_80.rename(columns={"Average_Spend":"Average_Spend_>80 ($)"})
    df_avg_spend = pd.merge(
        df_spend_less_80, df_spend_more_80, on=['Category']
    )
    last_two_cols = df_avg_spend.columns[-2:]
    st.dataframe(
        df_avg_spend.style.format(
            {col: "{:.0f}" for col in last_two_cols}
        ),
        hide_index=True
    )
    st.write("There are many evidence show that the potential strategic customer is high age group.")
    with st.expander("By Age (≤80)"):
        pipeline.chart_customer_purchasing(df_use_less_80)
    with st.expander("By Age (>80)"):
        pipeline.chart_customer_purchasing(df_use_more_80)
    st.divider()

# =========================
# Membership
# =========================
with tab4:
    st.write("Observe customer active year and preference channel.")
    st.subheader("Customer membership & activity")
    pipeline.chart_customer_membership(df)
    st.divider()

# =========================
# Recency
# =========================
with tab5:
    st.write("The colour density by bar and box-plot are the same.")
    st.write("The darker, the age group more recency.")
    st.write("The average is then support by box-plot")
    st.subheader("Customer recency")
    with st.expander("By Age (≤80)"):
        df_use = df[df["Age"] <= 80]
        pipeline.chart_customer_recency(df_use)
    with st.expander("By Age (>80)"):
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