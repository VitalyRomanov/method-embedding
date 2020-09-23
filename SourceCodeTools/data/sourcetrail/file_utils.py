from csv import QUOTE_NONNUMERIC
import pandas as pd

def write_dataframe(df, out_path, **kwargs):
    df.to_csv(out_path, index=False, compression="bz2", quoting=QUOTE_NONNUMERIC, **kwargs)

def read_dataframe(path, **kwargs):
    pd.read_csv(path, quotechar = '"', escapechar = '\\', **kwargs)