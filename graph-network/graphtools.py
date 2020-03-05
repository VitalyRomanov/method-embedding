import dgl
import numpy as np

def compact_prop(df, prop):
    uniq = df[prop].unique()
    prop2pid = dict(zip(uniq, range(uniq.size)))
    compactor = lambda type: prop2pid[type]
    df['compact_'+prop] = df[prop].apply(compactor)
    return df

def create_hetero_graph(nodes, edges):

    def add_typed_ids(nodes):
        nodes = nodes.copy()

        nodes = compact_prop(nodes, 'type')
        nodes['type'] = 0

        nodes['typed_id'] = None
        uniq_types = nodes['type'].unique()
        type_counts = dict()

        llabels = {}

        id_maps = {}

        for type_id, type in enumerate(uniq_types):
            nodes_of_type = nodes[nodes['type'] == type].shape[0]
            nodes.loc[nodes[nodes['type'] == type].index, 'typed_id'] = list(range(nodes_of_type))
            type_counts[str(type)] = nodes_of_type

            id_maps[str(type)] = dict(
                zip(
                    nodes.loc[nodes[nodes['type'] == type].index, 'id'].values,
                    nodes.loc[nodes[nodes['type'] == type].index, 'typed_id'].values
                )
            )

            # TODO
            # node types as labels
            llabels[str(type)] = nodes.loc[nodes[nodes['type'] == type].index, 'compact_type'].values

        assert any(nodes['typed_id'].isna()) == False

        return nodes, type_counts, llabels, id_maps

    def assess_need_for_self_loops(nodes, edges):
        need_self_loop = set(edges['src'].values.tolist()) - set(edges['dst'].values.tolist())
        for nid in need_self_loop:
            edges = edges.append({
                "id": -1,
                "type": 99,
                "src": nid,
                "dst": nid
            }, ignore_index=True)

        return nodes, edges

    # TODO
    # this is a hack when where are only outgoing connections from this node type
    # nodes, edges = assess_need_for_self_loops(nodes, edges)

    nodes, node_type_counts, llabels, id_maps = add_typed_ids(nodes)

    node2gid = dict(zip(nodes['id'].values, nodes['typed_id'].values))

    node2type = dict(zip(nodes['id'].values, nodes['type'].values))

    type_lookup = lambda id: node2type[id]
    gid_lookup = lambda id: node2gid[id]

    edges['src_type'] = edges['src'].apply(type_lookup)
    edges['dst_type'] = edges['dst'].apply(type_lookup)

    possible_edges = edges.drop_duplicates(['src_type', 'type', 'dst_type'])[['src_type', 'type', 'dst_type']]

    subsets = {}
    # subsets_d = {}

    for ind, row in possible_edges.iterrows():
        subset_desc = (str(row.src_type), str(row.type), str(row.dst_type))
        subset = edges.query('src_type == %s and type==%s and dst_type==%s' % subset_desc)
        subsets[(subset_desc)] = list(
            zip(
                map(gid_lookup, subset['src'].values),
                map(gid_lookup, subset['dst'].values)
            )
        )



        # subsets_d[(subset_desc)] = list(
        #     zip(
        #         subset['src'].values,
        #         subset['dst'].values
        #     )
        # )

    from pprint import pprint
    # pprint(subsets)
    # pprint(subsets_d)
    # pprint(node_type_counts)

    # uniq_node_types = nodes['type'].unique()
    # type_map = dict(zip(uniq_node_types, range(len(uniq_node_types))))
    # labels = nodes['type'].apply(lambda type: type_map[type]).values

    g = dgl.heterograph(subsets, node_type_counts)

    labels = np.concatenate([llabels[ntype] for ntype in g.ntypes])

    return g, labels, id_maps #nodes #, llabels

def create_graph(nodes, edges):

    nodes = nodes.copy()
    edges = edges.copy()

    nodes['new_id'] = list(range(nodes.shape[0]))

    node_id_map = dict(zip(nodes['id'].values, nodes['new_id'].values))

    id2new = lambda id: node_id_map[id]

    edges['new_src'] = edges['src'].apply(id2new)
    edges['new_dst'] = edges['dst'].apply(id2new)

    g = dgl.DGLGraph()
    g.add_nodes(nodes.shape[0])
    g.add_edges(edges['new_src'].values.tolist(), edges['new_dst'].values.tolist())

    uniq_node_types = nodes['type'].unique()
    type_map = dict(zip(uniq_node_types, range(len(uniq_node_types))))
    labels = nodes['type'].apply(lambda type: type_map[type]).values

    return g, labels, node_id_map

class Embedder:
    def __init__(self, id_map, embeddings):
        self.e = embeddings
        self.ind = id_map
        aid, iid = zip(*id_map.items())
        self.inv = dict(zip(iid, aid))

    def __getitem__(self, key):
        return self.e[self.ind[key], :]