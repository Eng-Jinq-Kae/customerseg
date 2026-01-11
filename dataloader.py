import glob
import pandas as pd

def ref_col_dataset():
    ref_col = {
        'ID': 'Customer unique ID',
        'Year_Birth': 'Customer birth year (eg. 1963)',
        'Education': 'Customer education level',
        'Marital_Status': 'Customer marital status',
        'Income': 'Customer income level',
        'Kidhome': 'Number of kids in the household',
        'Teenhome': 'Number of teenagers in the household',
        'Dt_Customer': 'Date of enrolment with the supermarket',
        'Recency': 'Number of days since the last purchase',
        'MntX': 'X could be multiple rows of category of goods',
        'NumYPurchases': 'Y could be multiple rows of channels customer make purchase',
        'NumWebVisitsMonth': 'Number of visits to the websites in the last month'
    }
    df_ref_col = pd.DataFrame.from_dict(
        ref_col,
        orient="index",
        columns=["Description"]
    ).reset_index(names="Category")
    return df_ref_col


def read_excel_preprocess(df):
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
    return df


def read_excel_file():
    try:
        file = 'W33836-XLS-ENG.xlsx'
        df = pd.read_excel(file, sheet_name='marketing')
    except:
        # find any xlsx file in current directory
        xlsx_files = glob.glob("*.xlsx")
        if not xlsx_files:
            raise FileNotFoundError("No .xlsx files found in directory")
        # read the first xlsx file using Sheet1
        df = pd.read_excel(xlsx_files[0], sheet_name='Sheet1')
    df = read_excel_preprocess(df)
    return df