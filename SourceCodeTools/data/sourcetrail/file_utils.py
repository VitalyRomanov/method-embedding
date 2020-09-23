from csv import QUOTE_NONNUMERIC

def write_dataframe(df, out_path, mode="w", header=True):
    df.to_csv(out_path, mode=mode, index=False, header=header, compression="bz2", quoting=QUOTE_NONNUMERIC)