import sys
import pandas

for filename in sys.argv[1:]:
    print(f"Reading from {filename}")
    data = pandas.read_csv(filename)
    storename = filename.replace(".csv",".bz2")
    print(f"Compressing to {storename}")
    data.to_pickle(storename)