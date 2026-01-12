import streamlit as st
import pandas as pd
import dataloader as dl
from pathlib import Path
from io import StringIO

def helptext():
    df_ref_col = dl.ref_col_dataset()
    st.divider()
    st.markdown('''
                1. If you are uploading Excel, this app only read one sheet with name marketing.
                2. Upload button at the bottom.
                3. If the uploaded is read correctly, you will able to see green success at the bottom.
                4. Some columns you need to have in the dataset in order to run this supermarket analysis.
    ''')
    st.subheader("Column Guideline")
    st.dataframe(df_ref_col, hide_index=True)
    st.divider()


def page_link():
    # st.write("GOTO")
    st.page_link(
        "pages/1_Analysis.py",
        label="Go to Analysis to know about your data",
        icon=":material/eye_tracking:"
    )

if __name__ == "__main__":
    st.title("Nata Supermarkets: Customer Analytics")
    st.set_page_config(layout="wide")
    helptext()
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        file_ext = Path(uploaded_file.name).suffix.lower()
        if file_ext == ".csv":
            df = pd.read_csv(uploaded_file)
        elif file_ext in [".xls", ".xlsx"]:
            try:
                df = pd.read_excel(uploaded_file, sheet_name="marketing")
            except:
                df = pd.read_excel(uploaded_file, sheet_name="Sheet1")
        else:
            raise ValueError("Unsupported file format")
        
        valid_df, error_msg = dl.check_col_df(df)
        if valid_df:
            st.session_state.file_upload = df
            st.markdown(":material/check_circle: File uploaded.")
            df = st.session_state.file_upload

            if len(df) >= 5:
                # st.dataframe(df.head())
                st.success("Data completely loaded.")
                st.divider()
                page_link()

        else:
            st.warning("Column mismatch, please col guideline above.")
            st.write(f"Error: {error_msg}")
            st.stop()