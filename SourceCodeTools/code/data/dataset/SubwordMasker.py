import pandas as pd


class SubwordMasker:
    """
    Masker that tells which node ids are subwords for given nodes.
    """
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, **kwargs):
        self.instantiate(nodes, edges, **kwargs)

    def instantiate(self, nodes, edges, **kwargs):
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

        for node_type, dst_id, src_type, src_typed_id in edges[["dst_type", "dst_typed_id", "src_type", "src_typed_id"]].values:
            key = (node_type, dst_id)
            if key in self.lookup:
                self.lookup[key].append((src_type, src_typed_id))
            else:
                self.lookup[key] = [(src_type, src_typed_id)]

    def get_mask(self, ids):
        """
        Accepts node ids that represent embeddable tokens as an input
        :param ids:
        :return:
        """
        if isinstance(ids, dict):
            for_masking = dict()
            for node_type, typed_ids in ids.items():
                for typed_id in typed_ids:
                    for src_type_, src_id in self.lookup[(node_type, typed_id)]:
                        if src_type_ in for_masking:
                            for_masking[src_type_].add(src_id)
                        else:
                            for_masking[src_type_] = set(src_id)
        else:
            for_masking = set()
            for typed_id in ids:
                for src_type_, src_id in self.lookup[("node_", typed_id)]:
                    for_masking.add(src_id)
            for_masking = {"node_": list(for_masking)}

        return for_masking


class NodeNameMasker(SubwordMasker):
    """
    Masker that tells which node ids are subwords for variables mentioned in a given function.
    """
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, node2name: pd.DataFrame, tokenizer_path):
        super(NodeNameMasker, self).__init__(nodes, edges, node2name=node2name, tokenizer_path=tokenizer_path)

    def instantiate(self, nodes, orig_edges, **kwargs):
        edges = orig_edges.copy()
        if "type_backup" in edges.columns:
            type_col = "type_backup"
        elif "type" in edges.columns:
            type_col = "type"
        else:
            raise Exception("Column `type` or `backup_type` not found")

        from SourceCodeTools.nlp.embed.bpe import load_bpe_model, make_tokenizer
        tokenize = make_tokenizer(load_bpe_model(kwargs['tokenizer_path']))

        subword_nodes = nodes.query("type_backup == 'subword'")
        subword2key = dict(zip(subword_nodes["name"], zip(subword_nodes["type"], subword_nodes["typed_id"])))

        node2type = dict(zip(nodes["id"], nodes["type"]))
        node2typed_id = dict(zip(nodes["id"], nodes["typed_id"]))

        node2name = kwargs["node2name"]
        node2name.eval("src_type = src.map(@node2type.get)", local_dict={"node2type": node2type}, inplace=True)
        node2name.eval("src_typed_id = src.map(@node2typed_id.get)", local_dict={"node2typed_id": node2typed_id},
                   inplace=True)

        self.lookup = {}
        for node_type, node_id, var_name in node2name[["src_type", "src_typed_id", "dst"]].values:
            key = (node_type, node_id)
            subwords = tokenize(var_name)
            if key not in self.lookup:
                self.lookup[key] = []

            # TODO
            #  Some subwords did not appear in the list of known subwords. Although this is not an issue,
            #  this can indicate that variable names are not extracted correctly. Need to verify.
            self.lookup[key].extend([subword2key[sub] for sub in subwords if sub in subword2key])


class NodeClfMasker(SubwordMasker):
    """
    Masker that tells which node ids are subwords for variables mentioned in a given function.
    """
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, **kwargs):
        super(NodeClfMasker, self).__init__(nodes, edges, **kwargs)

    def instantiate(self, nodes, orig_edges, **kwargs):
        pass

    def get_mask(self, ids):
        """
        Accepts node ids that represent embeddable tokens as an input
        :param ids:
        :return:
        """
        if isinstance(ids, dict):
            for_masking = ids
        else:
            for_masking = {"node_": ids}

        return for_masking
