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

def check_col_df(df):
    valid_df = True
    error_msg = None
    cols_df = df.columns
    if 'ID' not in cols_df:
        valid_df = False
        error_msg = 'Missing ID'
        return valid_df, error_msg
    if 'Year_Birth' not in cols_df:
        valid_df = False
        error_msg = 'Missing Year_Birth'
        return valid_df, error_msg
    if 'Income' not in cols_df:
        valid_df = False
        error_msg = 'Missing Income'
        return valid_df, error_msg
    if 'Dt_Customer' not in cols_df:
        valid_df = False
        error_msg = 'Missing Dt_Customer'
        return valid_df, error_msg
    mnt_cols = [c for c in df.columns if c.startswith("Mnt")]
    if len(mnt_cols) == 0:
        valid_df = False
        error_msg = 'Missing Mnt'
        return valid_df, error_msg
    return valid_df, error_msg


def read_df_preprocess(df):
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
        try:
            df = pd.read_excel(xlsx_files[0], sheet_name="marketing")
        except:
            df = pd.read_excel(xlsx_files[0], sheet_name="Sheet1")
    df = read_df_preprocess(df)
    return df