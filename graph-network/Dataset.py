import pandas
import dgl
import numpy

# TODO
# create model where variable names are in the graph. This way we can feed
# information about ast inside he function embedding


def load_data(node_path, edge_path):

    nodes = pandas.read_csv(node_path)
    edges = pandas.read_csv(edge_path)

    nodes_ = nodes.rename(mapper={
        'serialized_name': 'name'
    }, axis=1)

    edges_ = edges.rename(mapper={
        'source_node_id': 'src',
        'target_node_id': 'dst'
    }, axis=1)

    return nodes_, edges_


# def compact_prop(df, prop):
#     uniq = df[prop].unique()
#     prop2pid = dict(zip(uniq, range(uniq.size)))
#     compactor = lambda type: prop2pid[type]
#     df['compact_' + prop] = df[prop].apply(compactor)
#     return df

def compact_property(values):
    uniq = numpy.unique(values)
    prop2pid = dict(zip(uniq, range(uniq.size)))
    return prop2pid


def get_train_test_val_indices(labels):
    numpy.random.seed(42)

    indices = numpy.arange(start=0, stop=labels.size)
    numpy.random.shuffle(indices)

    train = int(indices.size * 0.6)
    test = int(indices.size * 0.7)

    return indices[:train], indices[train: test], indices[test:]


class SourceGraphDataset:
    def __init__(self, nodes_path, edges_path,
                 label_from, node_types=False,
                 edge_types=False, filter=None):
        # TODO
        # implemet filtering, so that when you filter a package
        # isolated nodes drop

        # self.nodes = pandas.read_csv(nodes_path)
        # self.edges = pandas.read_csv(edges_path)

        self.nodes, self.edges = load_data(nodes_path, edges_path)

        self.nodes, self.edges, self.held = SourceGraphDataset.holdout(self.nodes, self.edges)

        self.nodes_have_types = node_types
        self.edges_have_types = edge_types
        self.labels_from = label_from

        self.g = None


        self.nodes['label'] = self.nodes[label_from]
        self.label_map = compact_property(self.nodes['label'])

        assert any(pandas.isna(self.nodes['label'])) == False

        if not self.nodes_have_types:
            self.nodes['type'] = 0

        self.create_graph()
        self.splits = get_train_test_val_indices(self.labels)

    @property
    def labels(self):
        compact_label = lambda lbl: self.label_map[lbl]

        if self.edges_have_types:
            typed_labels = self.typed_labels
            return numpy.concatenate(
                [
                    numpy.array([compact_label(lbl) for lbl in typed_labels[ntype]])
                    for ntype in self.g.ntypes
                ]
            )
        else:
            return self.nodes['label'].apply(compact_label).values

    @property
    def typed_labels(self):
        typed_labels = dict()

        unique_types = self.nodes['type'].unique()

        for type_id, type in enumerate(unique_types):
            nodes_of_type = self.nodes[self.nodes['type'] == type]

            typed_labels[str(type)] = self.nodes.loc[
                nodes_of_type.index, 'label'
            ].values

        return typed_labels


    # @property
    # def node_id_maps(self):
    #     if self.node_id_map_ is not None:
    #         if self.edges_have_types:
    #             orig_id = []
    #             graph_id = []
    #             prev_offset = 0
    #
    #             for type in self.g.ntypes:
    #                 from_id, to_id = zip(*self.node_id_map_[type].items())
    #                 orig_id.extend(from_id)
    #                 graph_id.extend([t + prev_offset for t in to_id])
    #                 prev_offset += self.g.number_of_nodes(type)
    #
    #             return dict(zip(orig_id, graph_id))
    #         else:
    #             return self.node_id_map_
    #     else:
    #         return None

    @property
    def num_classes(self):
        return numpy.unique(self.labels).size

    def create_graph(self):
        if not self.nodes_have_types and not self.edges_have_types:
            self.create_directed_graph()
        elif self.edges_have_types:
            self.create_hetero_graph(
                node_types=self.nodes_have_types
            )
        else:
            raise NotImplemented("Edges should have type")

    @property
    def typed_node_id_maps(self):
        typed_id_maps = dict()

        unique_types = self.nodes['type'].unique()

        for type_id, type in enumerate(unique_types):
            nodes_of_type = self.nodes[self.nodes['type'] == type]
            typed_node_id2graph_id = compact_property(
                self.nodes.loc[nodes_of_type.index, 'id'].values
            )

            typed_id_maps[str(type)] = typed_node_id2graph_id

        return typed_id_maps

    @property
    def node_id_map(self):
        if self.edges_have_types:
            orig_id = []; graph_id = []; prev_offset = 0

            typed_node_id_maps = self.typed_node_id_maps

            for type in self.g.ntypes:
                from_id, to_id = zip(*typed_node_id_maps[type].items())
                orig_id.extend(from_id)
                graph_id.extend([t + prev_offset for t in to_id])
                prev_offset += self.g.number_of_nodes(type)

            return dict(zip(orig_id, graph_id))

        else:
            return compact_property(self.nodes['id'].values)


    @property
    def typed_node_counts(self):

        typed_node_counts = dict()

        unique_types = self.nodes['type'].unique()

        for type_id, type in enumerate(unique_types):
            nodes_of_type = self.nodes[self.nodes['type'] == type]
            typed_node_counts[str(type)] = nodes_of_type.shape[0]

        return typed_node_counts

    def create_directed_graph(self):
        nodes = self.nodes.copy()
        edges = self.edges.copy()

        node_id2graph_id = self.node_id_map

        id2new = lambda id: node_id2graph_id[id]

        graph_src = edges['src'].apply(id2new).values.tolist()
        graph_dst = edges['dst'].apply(id2new).values.tolist()

        g = dgl.DGLGraph()
        g.add_nodes(nodes.shape[0])
        g.add_edges(graph_src, graph_dst)

        self.g = g

    def create_hetero_graph(self, node_types=False, edge_types=False):
        # TODO
        # arguments are still not used

        nodes = self.nodes.copy()
        edges = self.edges.copy()

        # TODO
        # this is a hack when where are only outgoing connections from this node type
        # nodes, edges = Dataset.assess_need_for_self_loops(nodes, edges)

        typed_node_id_maps = self.typed_node_id_maps

        node2type = dict(zip(nodes['id'].values, nodes['type'].values))

        def graphid_lookup(nid):
            for type, maps in typed_node_id_maps.items():
                if nid in maps:
                    return maps[nid]

        type_lookup = lambda id: node2type[id]

        edges['src_type'] = edges['src'].apply(type_lookup)
        edges['dst_type'] = edges['dst'].apply(type_lookup)

        possible_edge_signatures = edges[['src_type', 'type', 'dst_type']].drop_duplicates(
            ['src_type', 'type', 'dst_type']
        )

        # typed_subgraphs is a dictionary with subset_signature as a key,
        # the dictionary stores directed edge lists
        typed_subgraphs = {}

        for ind, row in possible_edge_signatures.iterrows():
            subgraph_signature = (str(row.src_type), str(row.type), str(row.dst_type))

            subset = edges.query('src_type == %s and type==%s and dst_type==%s' % subgraph_signature)

            typed_subgraphs[(subgraph_signature)] = list(
                zip(
                    map(graphid_lookup, subset['src'].values),
                    map(graphid_lookup, subset['dst'].values)
                )
            )

        self.g = dgl.heterograph(typed_subgraphs, self.typed_node_counts)

        # self.typed_labels_ = typed_labels
        # self.typed_id_maps_ = typed_id_maps

        # self.labels_ = numpy.concatenate([typed_labels[ntype] for ntype in self.g.ntypes])
        #
        # orig_id = []
        # graph_id = []
        # prev_offset = 0
        #
        # for type in self.g.ntypes:
        #     from_id, to_id = zip(*typed_id_maps[type].items())
        #     orig_id.extend(from_id)
        #     graph_id.extend([t + prev_offset for t in to_id])
        #     prev_offset += self.g.number_of_nodes(type)
        #
        # self.node_id_map = dict(zip(orig_id, graph_id))

    # def add_typed_ids(self, nodes):
    #     nodes = nodes.copy()
    #
    #     nodes = compact_prop(nodes, 'type')
    #     nodes['type'] = 0
    #
    #     nodes['typed_id'] = None
    #     uniq_types = nodes['type'].unique()
    #     type_counts = dict()
    #
    #     typed_labels = {}
    #
    #     typed_id_maps = {}
    #
    #     for type_id, type in enumerate(uniq_types):
    #         nodes_of_type = nodes[nodes['type'] == type].shape[0]
    #         nodes.loc[nodes[nodes['type'] == type].index, 'typed_id'] = list(range(nodes_of_type))
    #         type_counts[str(type)] = nodes_of_type
    #
    #         typed_id_maps[str(type)] = dict(
    #             zip(
    #                 nodes.loc[nodes[nodes['type'] == type].index, 'id'].values,
    #                 nodes.loc[nodes[nodes['type'] == type].index, 'typed_id'].values
    #             )
    #         )
    #
    #         # TODO
    #         # node types as labels
    #         typed_labels[str(type)] = nodes.loc[
    #             nodes[nodes['type'] == type].index, 'label'
    #         ].apply(
    #             lambda type: self.label_map[type]
    #         ).values
    #
    #     assert any(nodes['typed_id'].isna()) == False
    #
    #     return nodes, type_counts, typed_labels, typed_id_maps

    @classmethod
    def assess_need_for_self_loops(cls, nodes, edges):
        need_self_loop = set(edges['src'].values.tolist()) - set(edges['dst'].values.tolist())
        for nid in need_self_loop:
            edges = edges.append({
                "id": -1,
                "type": 99,
                "src": nid,
                "dst": nid
            }, ignore_index=True)

        return nodes, edges

    @classmethod
    def holdout(cls, nodes, edges):
        train, test = split(edges)

        nodes, train_edges = ensure_connectedness(nodes, train)

        nodes, test_edges = ensure_valid_edges(nodes, test)

        return nodes, train_edges, test_edges




def split(edges):
    edges_shuffled = edges.sample(frac=1., random_state=42)

    # TODO
    # 0.9 is too much, need much less for evaluation
    train_frac = int(edges_shuffled.shape[0] * 0.9)
    # print("Current train frac: ", train_frac)

    train = edges_shuffled \
                .iloc[:train_frac]
    test = edges_shuffled \
               .iloc[train_frac:]
    return train, test


def ensure_connectedness(nodes, edges):
    unique_nodes = set(edges['src'].values.tolist() +
                       edges['dst'].values.tolist())

    nodes = nodes[
        nodes['id'].apply(lambda nid: nid in unique_nodes)
    ]

    return nodes, edges

def ensure_valid_edges(nodes, edges):
    unique_nodes = set(nodes['id'].values.tolist())

    edges = edges[
        edges['src'].apply(lambda nid: nid in unique_nodes)
    ]

    edges = edges[
        edges['dst'].apply(lambda nid: nid in unique_nodes)
    ]

    return nodes, edges