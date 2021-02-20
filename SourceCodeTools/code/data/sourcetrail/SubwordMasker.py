import pandas as pd

class SubwordMasker:
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame):
        edges = edges.copy()
        if "type_backup" in edges.columns:
            type_col = "type_backup"
        elif "type" in edges.columns:
            type_col = "type"
        else:
            raise Exception("Column `type` or `backup_type` not found")

        edges = edges.query(f"{type_col} == 'subword_'")
        self.lookup = dict()

        node2type = dict(zip(nodes["id"], nodes["type"]))
        node2typed_id = dict(zip(nodes["id"], nodes["typed_id"]))
        edges.eval("dst_type = dst.map(@node2type.get)", local_dict={"node2type": node2type}, inplace=True)
        edges.eval("dst_typed_id = dst.map(@node2typed_id.get)", local_dict={"node2typed_id": node2typed_id}, inplace=True)
        edges.eval("src_type = src.map(@node2type.get)", local_dict={"node2type": node2type}, inplace=True)
        edges.eval("src_typed_id = src.map(@node2typed_id.get)", local_dict={"node2typed_id": node2typed_id},
                   inplace=True)

        for dst_type, dst_edges in edges.groupby(by="dst_type"):
            self.lookup[dst_type] = dict()
            for dst_id, dst_id_dst_type_edges in dst_edges.groupby(by="dst_typed_id"):
                self.lookup[dst_type][dst_id] = dict()
                for src_type, src_type_dst_id_dst_type_edges in dst_id_dst_type_edges.groupby(by="src_type"):
                    self.lookup[dst_type][dst_id][src_type] = src_type_dst_id_dst_type_edges["src_typed_id"].to_list()

    def get_mask(self, ids):
        if isinstance(ids, dict):
            for_masking = dict()
            for node_type, typed_ids in ids.items():
                for typed_id in typed_ids:
                    for src_type_, src_id in self.lookup[node_type][typed_id].items():
                        if src_type_ in for_masking:
                            for_masking[src_type_].update(src_id)
                        else:
                            for_masking[src_type_] = set(src_id)
        else:
            for_masking = set()
            assert len(self.lookup) == 1
            node_type = next(iter(self.lookup.keys()))
            for typed_id in ids:
                for src_type_, src_id in self.lookup[node_type][typed_id].items():
                    for_masking.update(src_id)
            for_masking = {"node_": for_masking}

        return for_masking