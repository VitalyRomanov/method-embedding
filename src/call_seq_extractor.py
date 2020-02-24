#%%
 
import ast
import sys
import pandas

bodies_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/function_bodies/functions.csv" # sys.argv[1]
nodes_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/normalized_sourcetrail_nodes.csv" # sys.argv[2]
edges_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/component_0/edges_component_0.csv" # sys.argv[3]
occurrence_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/occurrence.csv"
source_location_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/source_location.csv"

#%%

bodies = pandas.read_csv(bodies_path)
nodes = pandas.read_csv(nodes_path)
edges = pandas.read_csv(edges_path)
occurrence = pandas.read_csv(occurrence_path)
source_location = pandas.read_csv(source_location_path)

#%% 

# occurrence.set_index("element_id", inplace=True)
# source_location.set_index("id", inplace=True)
# occurrence.sort_index(inplace=True)
# source_location.sort_index(inplace=True)

#%%

src_loc_id2data = zip(
        source_location['id'].values,
        zip(
            source_location['file_node_id'].values, 
            source_location['start_line'].values, 
            source_location['end_line'].values
            )
        )


#%%

nodes_id2name = dict(zip(nodes['id'].values, nodes['serialized_name'].apply(lambda x: x.split(".")[-1]).values))

# nodes_id2name = dict()

# for ind, row in nodes.iterrows():
#     nodes_id2name[row['id']] = row['serialized_name'].

#%%

edges_group = edges[edges['type'] == 8].groupby("source_node_id")


# %%

def get_body(node_id):
    try:
        return bodies.query(f"id == {node_id}").loc[0]['content']
    except KeyError:
        return None

def get_definition_location(node_id):
    locations_ids = occurrence.query(f"element_id == {node_id}")['source_location_id'].values
    if len(locations_ids) == 0: 
        return None
    definition = pandas.concat([source_location.query(f"id == {lid}") for lid in locations_ids], axis=0).query("type == 1")
    if definition.shape[0] == 0:
        return None
    return definition.iloc[0]

#%%

for ind, (node_id, group) in enumerate(edges_group):

    # body_str = get_body(node_id)
    chld = group['target_node_id'].values

    def_loc = get_definition_location(node_id)

    # if body_str is None and def_loc is not None or body_str is not None and def_loc is None:
    if def_loc is not None:
        print(node_id, chld, def_loc)

    if ind == 20: break

# %%
