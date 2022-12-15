from collections import defaultdict

import dgl
import numpy as np

from SourceCodeTools.code.annotator_utils import resolve_self_collisions2
from SourceCodeTools.code.data.ast_graph.build_ast_graph import ast_graph_for_single_example
from SourceCodeTools.code.data.dataset.Dataset import SimpleGraphCreator, SourceGraphDataset
from SourceCodeTools.models.training_config import get_config
from SourceCodeTools.nlp.batchers import PythonBatcher
from SourceCodeTools.nlp.batchers.PythonBatcher import MapperSpec


class HybridBatcher(PythonBatcher):
    def __init__(self, *args, **kwargs):
        dataset_config = get_config(data_path=kwargs["cache_dir"].parent)
        self._graph_creator = SourceGraphDataset(
            **dataset_config["DATASET"], **dataset_config["TOKENIZER"]
        )
        super(HybridBatcher, self).__init__(*args, **kwargs)

    def _prepare_tokenized_sent(self, sent):
        text, annotations = sent

        nodes, edges, offsets = ast_graph_for_single_example(text, "/Users/LTV/dev/method-embeddings/examples/sentencepiece_bpe.model", track_offsets=True)
        edges = edges.rename({"source_node_id": "src", "target_node_id": "dst"}, axis=1)
        nodes = nodes.rename({"serialized_name": "name"}, axis=1)
        graph = self._graph_creator.create_graph_from_nodes_and_edges(nodes, edges)

        doc = self._nlp(text)
        ents = annotations['entities']
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
        }

        output.update(self._parse_additional_tags(text, annotations, doc, output))

        return output

    def _create_additional_encoders(self):

        super()._create_additional_encoders()

        def graph_encoder(*args):
            graph = args[0]
            return graph

        self._mappers.append(
            MapperSpec(
                field="graph", target_field="graph", encoder=None, dtype=None, preproc_fn=None,
                encoder_fn=graph_encoder
            )
        )

    def format_batch(self, batch):
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