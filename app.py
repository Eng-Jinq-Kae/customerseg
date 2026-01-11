import streamlit as st
import pandas as pd
import dataloader as dl
from io import StringIO

def helptext():
    df_ref_col = dl.ref_col_dataset()
    st.divider()
    st.markdown('''
                1. If you are uploading Excel, this app only read one sheet with name Sheet1.
                2. Upload button at the bottom.
                3. If the uploaded is read correctly, you will able to see green success at the bottom.
                4. Some columns you need to run this supermarket analysis.
    ''')
    st.dataframe(df_ref_col, hide_index=True)
    st.divider()


def page_link():
    # st.write("GOTO")
    st.page_link(
        "pages/1_Describe.py",
        label="Go to Page Describe to know about your data",
        icon=":material/eye_tracking:"
    )
    st.page_link(
        "pages/2_Clustering.py",
        label="Go to Page Clustering",
        icon=":material/online_prediction:"
    )

if __name__ == "__main__":
    st.title("Nata Supermarkets: Customer Analytics")
    helptext()
    st.subheader("Upload your file here")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            dataframe = pd.read_csv(uploaded_file)
        except:
            dataframe = pd.read_excel(uploaded_file, sheet_name='Sheet1')
        st.session_state.file_upload = dataframe
    
        if 'file_upload' not in st.session_state or st.session_state.file_upload is None:
            st.markdown(":material/attach_file: No file uploaded, Reading data from directory.")
            df = dl.read_excel_file()
        else:
            st.markdown(":material/check_circle: File uploaded.")
            df = st.session_state.file_upload

        if len(df) >= 5:
            st.divider()
            # st.dataframe(df.head())
            st.success("Data completely loaded.")
            page_link()