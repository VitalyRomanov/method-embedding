#%%
import pandas as pd
import os, sys

nodes_path = sys.argv[1]
types_path = sys.argv[2]

nodes = pd.read_csv(nodes_path)
types = pd.read_csv(types_path)

#%%

types.rename({'src': 'dst', 'dst': 'src'}, axis=1, inplace=True)

nodes.rename({'id': 'dst'}, axis=1, inplace=True)

types_and_names = types.merge(nodes, on='dst')\
    .drop('dst', axis=1)\
    .rename({'serialized_name': 'dst'}, axis=1)[['src', 'dst']]

types_and_names.to_csv(os.path.join(os.path.dirname(types_path), "types_decoded.csv"), index=False)

