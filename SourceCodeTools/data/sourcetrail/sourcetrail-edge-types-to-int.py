import pandas as pd
import sys, os
from csv import QUOTE_NONNUMERIC

# def compact_prop(df, prop):
#     uniq = df[prop].unique()
#     prop2pid = dict(zip(uniq, range(-uniq.size, 0)))
#     compactor = lambda type: prop2pid[type]
#     df['compact_' + prop] = df[prop].apply(compactor)
#     return df

def isanumber(x):
    try:
        int(x)
        return True
    except:
        return False

common_edges = pd.read_csv(sys.argv[1])

uniq = common_edges['type'].unique()
uniq = [i for i in uniq if not isanumber(i)]
prop2pid = dict(zip(uniq, range(-len(uniq), 0)))

common_edges['type'] = common_edges['type'].apply(lambda x: prop2pid.get(x, x))

common_edges = common_edges.astype({"type": "int64"})

common_edges.to_csv(sys.argv[2], index=False, quoting=QUOTE_NONNUMERIC)

with open(sys.argv[3], "w") as sink:
    sink.write("type,desc\n")
    for item,val in prop2pid.items():
        sink.write(f"{val},{item}\n")