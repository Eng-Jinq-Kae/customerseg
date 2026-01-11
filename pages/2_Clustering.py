import streamlit as st
import dataloader as dl

st.title("Clustering")
if 'file_upload' not in st.session_state or st.session_state.file_upload is None:
    st.markdown(":material/attach_file: No file uploaded, Reading data from directory.")
    df = dl.read_excel_file()
else:
    st.markdown(":material/upload_file: Reading uplaoded file.")
    df = st.session_state.file_upload

st.dataframe(df.head())