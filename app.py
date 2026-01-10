import streamlit as st
import pandas as pd
import dataloader as dl
from io import StringIO

def page_link():
    st.write("GOTO")
    st.page_link(
        "pages/1_Describe.py",
        label="Go to Describe Page to know about your data",
        icon=":material/eye_tracking:"
    )
    st.page_link(
        "pages/2_Clustering.py",
        label="Go to Clustering Page",
        icon=":material/online_prediction:"
    )

if __name__ == "__main__":
    st.title("Nata Supermarkets: Customer Analytics")
    st.subheader("Upload your file here")
    page_link()

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.session_state.file_upload = dataframe
    
        if 'file_upload' not in st.session_state or st.session_state.file_upload is None:
            st.write("No file uploaded, Reading data from directory.")
            df = dl.read_excel_file()
        else:
            st.write("File uploaded is readed.")
            df = st.session_state.file_upload

        st.dataframe(df.head())
        page_link()
    
    st.success("Completely loaded.")