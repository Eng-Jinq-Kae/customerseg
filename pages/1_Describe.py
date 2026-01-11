import streamlit as st
import dataloader as dl
import pipeline as pipeline
import altair as alt

def chart_customer_age(df):
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X('Age:Q', bin=alt.Bin(maxbins=20), title='Age'),
            y=alt.Y('count()', title='Number of Customers')
        )
        .properties(title='Customer Age Distribution')
    )

    st.altair_chart(chart, use_container_width=True)

st.title("Describe")
if 'file_upload' not in st.session_state or st.session_state.file_upload is None:
    st.markdown(":material/attach_file: No file uploaded, Reading data from directory.")
    df = dl.read_excel_file()
else:
    st.markdown(":material/check_circle: Reading uplaoded file.")
    df = st.session_state.file_upload

# st.dataframe(df.head())
df = pipeline.add_customer_age(df)
df = pipeline.impute_missing_salary(df)
chart_customer_age(df)