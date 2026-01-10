import pandas as pd

def read_excel_file():
    file = 'W33836-XLS-ENG.xlsx'
    df = pd.read_excel(file, sheet_name='marketing')
    return df