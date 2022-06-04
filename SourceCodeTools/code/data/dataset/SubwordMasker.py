import pandas as pd
import torch

from SourceCodeTools.code.data.SQLiteStorage import SQLiteStorage

# TODO
# masking using edge weight
# https://discuss.dgl.ai/t/how-to-pass-in-edge-weights-in-heterographconv/2207

class SubwordMasker:
    """
    Masker that tells which node ids are subwords for given nodes.
    """
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, **kwargs):
        self.instantiate(nodes, edges, **kwargs)

    def instantiate(self, nodes, edges, **kwargs):
        # edges = edges.copy()
        # if "type_backup" in edges.columns:
        #     type_col = "type_backup"
        # elif "type" in edges.columns:
        #     type_col = "type"
        # else:
        #     raise Exception("Column `type` or `backup_type` not found")

        # nodes_table = SQLiteStorage(":memory:")
        # nodes_table.add_records(nodes, "nodes", create_index=["type_backup"])
        edges_table = SQLiteStorage(":memory:")
        edges_table.add_records(edges, "edges", create_index=["type_backup"])

        # edges = edges.query(f"{type_col} == 'subword_'")
        # edges = edges.query(f"type_backup == 'subword'")
        edges = edges_table.query(f"SELECT * FROM edges WHERE type_backup = 'subword'")
        self.lookup = dict()

        for dst_id, src_id in edges[["dst", "src"]].values:
            if dst_id in self.lookup:
                self.lookup[dst_id].append(src_id)
            else:
                self.lookup[dst_id] = [src_id]

        # node2type = dict(zip(nodes["id"], nodes["type"]))
        # # node2typed_id = dict(zip(nodes["id"], nodes["typed_id"]))
        # get_node_type = lambda x: node2type.get(id, pd.NA)
        # edges.eval("dst_type = dst.map(@node2type)", local_dict={"node2type": get_node_type}, inplace=True)
        # # edges.eval("dst_typed_id = dst.map(@node2typed_id.get)", local_dict={"node2typed_id": node2typed_id}, inplace=True)
        # edges.eval("src_type = src.map(@node2type)", local_dict={"node2type": get_node_type}, inplace=True)
        # # edges.eval("src_typed_id = src.map(@node2typed_id.get)", local_dict={"node2typed_id": node2typed_id}, inplace=True)

        # for node_type, dst_id, src_type, src_typed_id in edges[["dst_type", "dst_typed_id", "src_type", "src_typed_id"]].values:
        #     key = (node_type, dst_id)
        #     if key in self.lookup:
        #         self.lookup[key].append((src_type, src_typed_id))
        #     else:
        #         self.lookup[key] = [(src_type, src_typed_id)]

    def apply_mask(self, input_nodes, for_masking):
        def create_mask(input_ids):
            return torch.BoolTensor(list(map(lambda id_: id_ not in for_masking, input_ids.tolist())))

        if isinstance(input_nodes, dict):
            mask = dict()
            for node_type, ids in input_nodes.items():
                mask[node_type] = create_mask(ids)
        else:
            mask = create_mask(input_nodes)

        return mask

    def get_mask(self, *, mask_for, input_nodes):
        """
        Accepts node ids that represent embeddable tokens as an input
        :param ids:
        :return:
        """
        for_masking = set()

        def get_masked_ids(mask_for, for_masking):
            for typed_id in mask_for:
                if typed_id in self.lookup:
                    for_masking.update(self.lookup[typed_id])

        if isinstance(mask_for, dict):
            for node_type, typed_ids in mask_for.items():
                get_masked_ids(typed_ids.tolist(), for_masking)
        else:
            get_masked_ids(mask_for.tolist(), for_masking)

        return self.apply_mask(input_nodes, for_masking)


class NodeNameMasker(SubwordMasker):
    """
    Masker that tells which node ids are subwords for variables mentioned in a given function.
    """
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, node2name: pd.DataFrame, tokenizer_path):
        super(NodeNameMasker, self).__init__(nodes, edges, node2name=node2name, tokenizer_path=tokenizer_path)

    def instantiate(self, nodes, edges, **kwargs):
        # edges = orig_edges.copy()
        # if "type_backup" in edges.columns:
        #     type_col = "type_backup"
        # elif "type" in edges.columns:
        #     type_col = "type"
        # else:
        #     raise Exception("Column `type` or `backup_type` not found")

        from SourceCodeTools.nlp import create_tokenizer
        tokenize = create_tokenizer("bpe", bpe_path=kwargs['tokenizer_path'])

        nodes_table = SQLiteStorage(":memory:")
        nodes_table.add_records(nodes, "nodes", create_index=["type_backup"])
        # edges_table = SQLiteStorage(":memory:")
        # edges_table.add_records(edges, "edges", create_index=["type_backup"])

        # subword_nodes = nodes.query("type_backup == 'subword'")
        subword_nodes = nodes_table.query("SELECT * FROM nodes WHERE type_backup = 'subword'")
        # subword2key = dict(zip(subword_nodes["name"], zip(subword_nodes["type"], subword_nodes["typed_id"])))
        subword2key = dict(zip(subword_nodes["name"], subword_nodes["id"]))

        # node2type = dict(zip(nodes["id"], nodes["type"]))
        # get_node_type = lambda x: node2type.get(id, pd.NA)
        # node2typed_id = dict(zip(nodes["id"], nodes["typed_id"]))

        node2name = kwargs["node2name"]
        # node2name.eval("src_type = src.map(@node2type)", local_dict={"node2type": get_node_type}, inplace=True)
        # node2name.dropna(inplace=True)
        # # node2name.eval("src_typed_id = src.map(@node2typed_id.get)", local_dict={"node2typed_id": node2typed_id}, inplace=True)

        self.lookup = {}
        for node_id, var_name in node2name[["src", "dst"]].values:
            subwords = tokenize(var_name)
            if node_id not in self.lookup:
                self.lookup[node_id] = []

            # TODO
            #  Some subwords did not appear in the list of known subwords. Although this is not an issue,
            #  this can indicate that variable names are not extracted correctly. Need to verify.
            self.lookup[node_id].extend([subword2key[sub] for sub in subwords if sub in subword2key])


class NodeClfMasker(SubwordMasker):
    """
    Masker that tells which node ids are subwords for variables mentioned in a given function.
    """
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, **kwargs):
        super(NodeClfMasker, self).__init__(nodes, edges, **kwargs)

    def instantiate(self, nodes, orig_edges, **kwargs):
        pass

    def get_mask(self, *, mask_for, input_nodes):
        """
        Accepts node ids that represent embeddable tokens as an input
        :param ids:
        :return:
        """
        for_masking = set()
        if isinstance(mask_for, dict):
            for node_type, typed_ids in mask_for.items():
                for_masking.update(typed_ids.tolist())
        else:
            for_masking.update(mask_for.tolist())

        return self.apply_mask(input_nodes, for_masking)
