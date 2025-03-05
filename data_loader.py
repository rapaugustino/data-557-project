import pandas as pd

def load_data(file_path: str = "salary.txt") -> pd.DataFrame:
    """
    Loads the salary dataset from 'salary.txt' using flexible whitespace separation.
    Returns a pandas DataFrame with columns:
    ['case','id','sex','deg','yrdeg','field','startyr','year','rank','admin','salary'].
    """
    df = pd.read_csv(
        file_path,
        sep=r'[\t\s]+',  # regex for any amount of whitespace
        header=0,
        engine='python'
    )
    return df