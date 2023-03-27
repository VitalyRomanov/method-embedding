from collections import defaultdict

import dgl
import numpy as np

from SourceCodeTools.code.annotator_utils import resolve_self_collisions2
from SourceCodeTools.code.data.ast_graph.build_ast_graph import ast_graph_for_single_example, source_code_to_graph
from SourceCodeTools.code.data.dataset.Dataset import SimpleGraphCreator, SourceGraphDataset, ProxyDataset
from SourceCodeTools.nlp import ShiftingValueEncoder
from SourceCodeTools.nlp.batchers import PythonBatcher
from SourceCodeTools.nlp.batchers.PythonBatcher import MapperSpec


class HybridBatcher(PythonBatcher):
    def __init__(self, *args, **kwargs):
        self._graph_config = kwargs.pop("graph_config")
        assert kwargs["cache_dir"] is not None
        self._graph_config["DATASET"]["data_path"] = kwargs["cache_dir"]
        super(HybridBatcher, self).__init__(*args, **kwargs)

    def _prepare_tokenized_sent(self, sent):
        text, annotations = sent

        graph = source_code_to_graph(
            text,
            variety="v2.5",
            bpe_tokenizer_path=self._graph_config["TOKENIZER"]["tokenizer_path"],
            reverse_edges=True,
            mention_instances=False, save_node_strings=True
        )
        nodes, edges, offsets = graph["nodes"], graph["edges"], graph["offsets"]

        edges = edges.rename({"source_node_id": "src", "target_node_id": "dst"}, axis=1)
        nodes = nodes.rename({"serialized_name": "name"}, axis=1)

        graph_creator = ProxyDataset(
            storage_kwargs={"nodes": nodes, "edges": edges},
            **self._graph_config["DATASET"], **self._graph_config["TOKENIZER"]
        )

        graph = graph_creator.create_graph()

        doc = self._nlp(text)
        ents = annotations.get('entities', [])
        annotations['replacements'] = resolve_self_collisions2(
            list(zip(offsets["start"], offsets["end"], offsets["node_id"]))
        )

        tokens = doc
        try:
            tokens = [t.text for t in tokens]
        except:
            pass

        ents_tags = self._biluo_tags_from_offsets(doc, ents, check_localization_parameter=True)
        assert len(tokens) == len(ents_tags)

        output = {
            "tokens": tokens,
            "tags": ents_tags,
            "graph": graph,
            "nodes": nodes,
            "edges": edges,
            "offsets": offsets
        }

        output.update(self._parse_additional_tags(text, annotations, doc, output))

        return output

    def _create_graph_encoder(self):
        graphmap_enc = ShiftingValueEncoder()
        # graphmap_enc.set_default(len(self._graphmap))
        self._mappers.append(
            MapperSpec(
                field="replacements", target_field="graph_ids", encoder=graphmap_enc,
                preproc_fn=self._strip_biluo_and_try_cast_to_int, dtype=np.int32
            )
        )
        self.graphpad = graphmap_enc.default

    def _create_additional_encoders(self):

        super()._create_additional_encoders()

        def graph_encoder(*args, **kwargs):
            graph = args[0]
            return graph

        self._mappers.append(
            MapperSpec(
                field="graph", target_field="graph", encoder=None, dtype=None, preproc_fn=None,
                encoder_fn=graph_encoder
            )
        )

    def collate(self, batch):
        fbatch = defaultdict(list)

        for sent in batch:
            for key, val in sent.items():
                fbatch[key].append(val)

        max_len = max(fbatch["lens"])

        batch_o = {}

        for field, items in fbatch.items():
            if field == "lens" or field == "label":
                batch_o[field] = np.array(items, dtype=np.int32)
            elif field == "id":
                batch_o[field] = np.array(items, dtype=np.int64)
            elif field == "tokens" or field == "replacements":
                batch_o[field] = items
            elif field == "graph":
                batch_o[field] = dgl.batch(items)
            else:
                batch_o[field] = np.stack(items)[:, :max_len]

        return batch_o