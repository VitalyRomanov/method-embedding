#%%
import pandas as pd
import os, sys

nodes_path = sys.argv[1]
types_path = sys.argv[2]

nodes = pd.read_csv(nodes_path).astype({"id": "int32", "serialized_name": "str"})
types = pd.read_csv(types_path).astype({"src": "int32", "dst": "int32"})

#%%

types.rename({'src': 'dst', 'dst': 'src'}, axis=1, inplace=True)

nodes.rename({'id': 'dst'}, axis=1, inplace=True)

types_and_names = types.merge(nodes, on='dst')\
    .drop('dst', axis=1)\
    .rename({'serialized_name': 'dst'}, axis=1)[['src', 'dst']]

types_and_names['dst'] = types_and_names['dst'].apply(lambda x: x.split("[")[0].split(".")[-1].strip("\""))

counts = types_and_names['dst'].value_counts()

types_and_names['counts'] = types_and_names['dst'].apply(lambda x: counts[x])
types_and_names = types_and_names.query("counts > 1")[['src','dst']]

types_and_names.to_csv(os.path.join(os.path.dirname(types_path), "types_decoded.csv"), index=False)

