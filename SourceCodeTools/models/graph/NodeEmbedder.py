from SourceCodeTools.nlp import token_hasher
import torch
import torch.nn as nn


class SimpleNodeEmbedder(nn.Module):
    def __init__(self, dataset, emb_size, dtype=None, n_buckets=500000, pretrained=None):
        super(SimpleNodeEmbedder, self).__init__()

        self.emb_size = emb_size
        self.dtype = dtype
        if dtype is None:
            self.dtype = torch.float32
        self.n_buckets = n_buckets

        self.buckets = None

        from SourceCodeTools.code.data.sourcetrail.sourcetrail_ast_edges import PythonSharedNodes

        leaf_types = PythonSharedNodes.shared_node_types

        if len(dataset.nodes.query("type_backup == 'subword'")) > 0:
            # some of the types should not be embedded if subwords were generated
            leaf_types = leaf_types - {"#attr#"}
            leaf_types = leaf_types - {"#keyword#"}

        nodes_with_embeddings = dataset.nodes[
            dataset.nodes['type_backup'].apply(lambda type_: type_ in leaf_types)
        ][['global_graph_id', 'typed_id', 'type', 'type_backup', 'name']]

        type_name = list(zip(nodes_with_embeddings['type_backup'], nodes_with_embeddings['name']))

        self.node_info = dict(zip(
            list(zip(nodes_with_embeddings['type'], nodes_with_embeddings['typed_id'])),
            type_name
        ))

        assert len(nodes_with_embeddings) == len(self.node_info)

        self.node_info_global = dict(zip(
            nodes_with_embeddings['global_graph_id'],
            type_name
        ))

        if pretrained is None:
            self._create_buckets()
        else:
            self._create_buckets_from_pretrained(pretrained)

    def _create_buckets(self):
        self.buckets = nn.Embedding(self.n_buckets + 1, self.emb_size, padding_idx=self.n_buckets)

    def _create_buckets_from_pretrained(self, pretrained):

        assert pretrained.n_dims == self.emb_size

        import numpy as np

        embs_init = np.random.randn(self.n_buckets, self.emb_size).astype(np.float32)

        for word in pretrained.keys():
            ind = token_hasher(word, self.n_buckets)
            embs_init[ind, :] = pretrained[word]

        from SourceCodeTools.code.python_tokens_to_bpe_subwords import python_ops_to_bpe

        def op_embedding(op_tokens):
            embedding = None
            for token in op_tokens:
                token_emb = pretrained.get(token, None)
                if embedding is None:
                    embedding = token_emb
                else:
                    embedding = embedding + token_emb
            return embedding

        for op, op_tokens in python_ops_to_bpe.items():
            op_emb = op_embedding(op_tokens)
            if op_emb is not None:
                op_ind = token_hasher(op, self.n_buckets)
                embs_init[op_ind, :] = op_emb

        weights_with_pad = torch.tensor(np.vstack([embs_init, np.zeros((1, self.emb_size), dtype=np.float32)]))

        self.buckets = nn.Embedding.from_pretrained(weights_with_pad, freeze=False, padding_idx=self.n_buckets)

    def _get_embedding_from_node_info(self, keys, node_info):
        idxs = []

        for key in keys:
            if key in node_info:
                real_type, name = node_info[key]
                idxs.append(token_hasher(name, self.n_buckets))
            else:
                idxs.append(self.n_buckets)

        return self.buckets(torch.LongTensor(idxs))

    def _get_embeddings_with_type(self, node_type, ids):
        type_ids = ((node_type, id_) for id_ in ids)
        return self._get_embedding_from_node_info(type_ids, self.node_info)

    def _get_embeddings_global(self, ids):
        return self._get_embedding_from_node_info(ids, self.node_info_global)

    def get_embeddings(self, node_type=None, node_ids=None):
        assert node_ids is not None
        if node_type is None:
            return self._get_embeddings_global(node_ids)
        else:
            return self._get_embeddings_with_type(node_type, node_ids)

    def forward(self, node_type=None, node_ids=None, train_embeddings=True):
        if train_embeddings:
            return self.get_embeddings(node_type, node_ids.tolist())
        else:
            with torch.set_grad_enabled(False):
                return self.get_embeddings(node_type, node_ids.tolist())

class NodeEmbedder(nn.Module):
    def __init__(self, dataset, emb_size, tokenizer_path, dtype=None, n_buckets=100000, pretrained=None):
        super(NodeEmbedder, self).__init__()

        self.emb_size = emb_size
        self.dtype = dtype
        if dtype is None:
            self.dtype = torch.float32
        self.n_buckets = n_buckets

        self.bpe_tokenizer = None
        self.op_tokenizer = None
        # self.graph_id_to_pretrained_name = None
        self.pretrained_name_to_ind = None
        self.pretrained_embeddings = None
        self.buckets = None

        from SourceCodeTools.code.data.sourcetrail.sourcetrail_ast_edges import SharedNodeDetector

        leaf_types = SharedNodeDetector.shared_node_types

        if len(dataset.nodes.query("type_backup == 'subword'")) > 0:
            # some of the types should not be embedded if subwords were generated
            leaf_types = leaf_types - {"#attr#"}
            leaf_types = leaf_types - {"#keyword#"}

        nodes_with_embeddings = dataset.nodes[
            dataset.nodes['type_backup'].apply(lambda type_: type_ in leaf_types)
        ][['global_graph_id', 'typed_id', 'type', 'type_backup', 'name']]

        type_name = list(zip(nodes_with_embeddings['type_backup'], nodes_with_embeddings['name']))

        self.node_info = dict(zip(
            list(zip(nodes_with_embeddings['type'], nodes_with_embeddings['typed_id'])),
            type_name
        ))

        assert len(nodes_with_embeddings) == len(self.node_info)

        self.node_info_global = dict(zip(
            nodes_with_embeddings['global_graph_id'],
            type_name
        ))

        # self._create_ops_tokenization(nodes_with_embeddings)
        self._create_buckets()

        if pretrained is not None:
            self._create_pretrained_embeddings(nodes_with_embeddings, pretrained)

        self._create_zero_embedding()
        self._init_tokenizer(tokenizer_path)

    def _create_zero_embedding(self):
        self.zero = torch.zeros((self.emb_size, ), requires_grad=False)

    def _create_pretrained_embeddings(self, nodes, pretrained):
        # self.graph_id_to_pretrained_name = dict(zip(nodes['global_graph_id'], nodes['name']))
        self.pretrained_name_to_ind = pretrained.ind
        embed = nn.Parameter(torch.tensor(pretrained.e, dtype=self.dtype))
        # nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(embed)
        self.pretrained_embeddings = embed

    def _create_ops_tokenization(self, nodes_with_embeddings):
        ops = nodes_with_embeddings.query("type_backup == 'Op'")
        from SourceCodeTools.code.python_tokens_to_bpe_subwords import op_tokenizer

        self.ops_tokenized = dict(zip(ops['name'], ops['name'].apply(op_tokenizer)))

    def _create_buckets(self):
        embed = nn.Parameter(torch.Tensor(self.n_buckets, self.emb_size))
        # nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(embed)
        self.buckets = embed

    def _init_tokenizer(self, tokenizer_path):
        from SourceCodeTools.nlp.embed.bpe import load_bpe_model, make_tokenizer
        self.bpe_tokenizer = make_tokenizer(load_bpe_model(tokenizer_path))
        from SourceCodeTools.code.python_tokens_to_bpe_subwords import op_tokenizer
        self.op_tokenizer = op_tokenizer

    def _tokenize(self, type_, name):
        tokenized = None
        if type_ == "Op":
            try_tokenized = self.op_tokenizer(name)
            if try_tokenized == name:
                tokenized = None

        if tokenized is None:
            tokenized = self.bpe_tokenizer(name)
        return tokenized

    def _get_pretrained_or_none(self, name):
        if self.pretrained_name_to_ind is not None and name in self.pretrained_name_to_ind:
            return self.pretrained_embeddings[self.pretrained_name_to_ind[name], :]
        else:
            return None

    def _get_from_buckets(self, name):
        return self.buckets[token_hasher(name, self.n_buckets), :]

    def _get_from_tokenized(self, type_, name):
        tokens = self._tokenize(type_, name)
        embedding = None
        for token in tokens:
            token_emb = self._get_pretrained_or_none(token)
            if token_emb is None:
                token_emb = self._get_from_buckets(token)

            if embedding is None:
                embedding = token_emb
            else:
                embedding = embedding + token_emb
        return embedding

    def _get_embedding(self, type_id, node_info):
        if type_id in node_info:
            real_type, name = self.node_info[type_id]
            embedding = self._get_pretrained_or_none(name)

            if embedding is None:
                embedding = self._get_from_tokenized(real_type, name)
        else:
            embedding = self.zero

        return embedding

    def _get_embeddings_with_type(self, node_type, ids):
        embeddings = []
        for id_ in ids:
            type_id = (node_type, id_)
            embeddings.append(self._get_embedding(type_id, self.node_info))
        embeddings = torch.stack(embeddings)
        return embeddings

    def _get_embeddings_global(self, ids):
        embeddings = []
        for global_id in ids:
            embeddings.append(self._get_embedding(global_id, self.node_info_global))
        embeddings = torch.stack(embeddings)
        return embeddings

    def get_embeddings(self, node_type=None, node_ids=None):
        assert node_ids is not None
        if node_type is None:
            return self._get_embeddings_global(node_ids)
        else:
            return self._get_embeddings_with_type(node_type, node_ids)

    def forward(self, node_type=None, node_ids=None, train_embeddings=True):
        if train_embeddings:
            return self.get_embeddings(node_type, node_ids.tolist())
        else:
            with torch.set_grad_enabled(False):
                return self.get_embeddings(node_type, node_ids.tolist())