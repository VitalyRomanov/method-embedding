import json
# import logging
from collections import defaultdict
# from enum import Enum
# from multiprocessing import Queue, Process
# from queue import Empty
# from time import sleep
# from typing import Union, Set

import numpy as np
from networkx import disjoint_union

from SourceCodeTools.code.annotator_utils import resolve_self_collisions2
from SourceCodeTools.code.data.DBStorage import Chunk
from SourceCodeTools.code.data.ast_graph.build_ast_graph import ast_graph_for_single_example, source_code_to_graph
from SourceCodeTools.code.data.dataset.Dataset import SimpleGraphCreator, SourceGraphDataset, ProxyDataset, \
    ProxyDatasetNoPandas
# from SourceCodeTools.code.data.multiprocessing_storage_adapter import Message
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
            variety="v3.5",
            bpe_tokenizer_path=self._graph_config["TOKENIZER"]["tokenizer_path"],
            reverse_edges=True,
            mention_instances=False, save_node_strings=True, make_table=False
        )
        nodes, edges, offsets = Chunk(graph["nodes"]), Chunk(graph["edges"]), Chunk(graph["offsets"])

        # edges = edges.rename({"source_node_id": "src", "target_node_id": "dst"}, axis=1)
        # nodes = nodes.rename({"serialized_name": "name"}, axis=1)

        def make_dense_encoding(hashes):
            dense = {}
            dense[None] = None
            for ind, h in enumerate(hashes):
                dense[h] = ind + 1
            return dense

        node2dense_id = make_dense_encoding(nodes["node_hash"])
        edge2dense_id = make_dense_encoding(edges["edge_hash"])

        nodes["id"] = map(node2dense_id.get, nodes["node_hash"])
        nodes["scope"] = map(node2dense_id.get, nodes["scope"])
        offsets["node_id"] = map(node2dense_id.get, offsets["node_id"])
        offsets["scope"] = map(node2dense_id.get, offsets["scope"])
        edges["id"] = map(edge2dense_id.get, edges["edge_hash"])
        edges["scope"] = map(node2dense_id.get, edges["scope"])
        edges["src"] = map(node2dense_id.get, edges["src"])
        edges["dst"] = map(node2dense_id.get, edges["dst"])
        # nodes["id"] = nodes["node_hash"]
        # edges["id"] = edges["edge_hash"]

        offsets["start"] = offsets["offset_start"]
        offsets["end"] = offsets["offset_end"]

        graph_creator = ProxyDatasetNoPandas(
            storage_kwargs={"nodes": nodes, "edges": edges},
            **self._graph_config["DATASET"], **self._graph_config["TOKENIZER"]
        )

        graph = graph_creator.create_graph()

        doc = self._nlp(text)
        ents = annotations.get('entities', [])
        # if "replacements" not in annotations: # don't do this for now because nodes from AST and global nodes overlap
        #     annotations["replacements"] = []
        # annotations['replacements'].extend(resolve_self_collisions2(
        #     list(zip(offsets["start"], offsets["end"], offsets["node_id"]))
        # ))
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

        # dgl_graph = dgl.from_networkx(graph, node_attrs=["original_id"], edge_attrs=["original_id"])
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
        if len(batch) == 0:
            return {}

        keys = list(batch[0].keys())
        keys.pop(keys.index("graph_ids"))
        keys.append("graph_ids")

        def get_key(key):
            for sent in batch:
                yield sent[key]

        max_len = min(max(get_key("lens")), self._max_seq_len)

        def add_padding(encoded, pad):
            blank = np.ones((max_len,), dtype=np.int32) * pad
            blank[0:min(encoded.size, max_len)] = encoded[0:min(encoded.size, max_len)]
            return blank

        def assign_ind_to_nodes(graph, ind):
            for node in graph:
                graph.nodes[node]["ind"] = ind

        batch_o = {}

        for field in keys:
            if field == "lens" or field == "label":
                batch_o[field] = np.fromiter((min(i, max_len) for i in get_key(field)), dtype=np.int32)
            elif field == "id":
                batch_o[field] = np.fromiter(get_key(field), dtype=np.int64)
            elif field == "tokens" or field == "replacements":
                batch_o[field] = get_key(field)
            elif field == "graph":
                graphs = list(get_key(field))
                united = None
                for ind, g in enumerate(graphs):
                    assign_ind_to_nodes(g, ind)
                    if united is None:
                        united = g
                    else:
                        united = disjoint_union(united, g)
                    # g.ndata["ind"] = torch.LongTensor([ind] * len(g.ndata["original_id"]))
                # batch_o[field] = dgl.batch(graphs)
                batch_o[field] = united
            elif field == "graph_ids":
                graph = batch_o["graph"]
                graph_id_mapping = defaultdict(dict)
                for graph_id, node in enumerate(graph):
                    node_data = graph.nodes[node]
                    graph_id_mapping[node_data["ind"]][node_data["original_id"]] = graph_id + 1
                # for graph_id, ind, original_id in zip(
                #         graph.nodes().tolist(),
                #         graph.ndata["ind"].tolist(),
                #         graph.ndata["original_id"].tolist()
                # ):
                #     graph_id_mapping[ind][original_id] = graph_id + 1   # need to do this because the padding is 0

                for ind in graph_id_mapping:
                    graph_id_mapping[ind][-1] = 0

                batch_o[field] = np.array(
                    [add_padding(
                        np.fromiter(map(lambda x: graph_id_mapping[ind][x-1], item), dtype=np.int32),
                        self._default_padding[field]
                    ) for ind, item in enumerate(get_key(field))],
                    dtype=np.int64
                )
            else:
                batch_o[field] = np.array(
                    [add_padding(item, self._default_padding[field]) for item in get_key(field)],
                    dtype=np.int64
                )

        return batch_o


class StreamIterator:
    def __init__(self, path):
        self.path = path
        self.count = None

    def __iter__(self):
        with open(self.path, "r") as source:
            for line in source:
                if line.strip():
                    data = json.loads(line)
                    data[1].pop("id")
                    yield data
                    # yield json.loads(line)

    def __len__(self):
        if self.count is None:
            count = 0
            with open(self.path, "r") as source:
                for line in source:
                    count += 1

            self.count = count
        return self.count


# class MPIteratorWorker:
#     class InboxTypes(Enum):
#         iterate = 0
#         get_len = 1
#
#     class OutboxTypes(Enum):
#         worker_started = 0
#         next = 1
#         stop_iteration = 2
#         len = 3
#
#     inbox_queue: Queue
#     outbox_queue: Queue
#     iteration_queue: Queue
#
#     def __init__(self, config, inbox_queue, outbox_queue, iteration_queue):
#         self._init(config)
#         self.inbox_queue = inbox_queue
#         self.outbox_queue = outbox_queue
#         self.iteration_queue = iteration_queue
#         self._send_init_confirmation()
#
#     def _init(self, config):
#         self.iter_fn = config.pop("iter_fn")
#         self.iter_fn_kwargs = config
#
#     def init_iterator(self):
#         return self.iter_fn(**self.iter_fn_kwargs)
#
#     def iter_len(self):
#         return len(self.iter_fn(**self.iter_fn_kwargs))
#
#     def _send_init_confirmation(self):
#         self.send_out(Message(
#             descriptor=MPIteratorWorker.OutboxTypes.worker_started,
#             content=None
#         ))
#
#     def send_out(self, message, queue=None, keep_trying=True) -> True:
#         """
#         :param message:
#         :param queue:
#         :param keep_trying: Block until can put the item in the queue
#         :return: Return True if could put item in the queue, and False otherwise
#         """
#         if queue is None:
#             queue = self.outbox_queue
#
#         if keep_trying:
#             while queue.full():
#                 sleep(0.2)
#         else:
#             if queue.full():
#                 return False
#
#         queue.put(message)
#         return True
#
#     def check_for_new_messages(self):
#         interrupt_iteration = False
#         try:
#             message = self.inbox_queue.get(timeout=0.0)
#             if message.descriptor == self.InboxTypes.iterate:
#                 while not self.outbox_queue.empty():
#                     self.outbox_queue.get()
#                 self.inbox_queue.put(message)
#                 interrupt_iteration = True
#             else:
#                 self._handle_message(message)
#         except Empty:
#             pass
#         return interrupt_iteration
#
#     def _iterate(self):
#         iterator = self.init_iterator()
#
#         for value in iterator:
#             sent = False
#             while not sent:
#                 interrupt_iteration = self.check_for_new_messages()
#                 if interrupt_iteration:
#                     return
#                 sent = self.send_out(Message(
#                     descriptor=self.OutboxTypes.next,
#                     content=value
#                 ), queue=self.iteration_queue, keep_trying=False)
#         self.send_out(Message(
#             descriptor=self.OutboxTypes.stop_iteration,
#             content=None
#         ), queue=self.iteration_queue)
#
#     def _handle_message(self, message):
#         if message.descriptor == self.InboxTypes.iterate:
#             self._iterate()
#
#         elif message.descriptor == self.InboxTypes.get_len:
#             self.send_out(Message(
#                 self.OutboxTypes.len, self.iter_len()
#             ))
#         else:
#             raise ValueError(f"Unrecognized message descriptor: {message.descriptor.name}")
#
#     def handle_incoming(self):
#         message = self.inbox_queue.get()
#         response = self._handle_message(message)
#
#
# def start_worker(config, inbox_queue, outbox_queue, iteration_queue, *args, **kwargs):
#     logging.basicConfig(level=logging.INFO,
#                         format="%(asctime)s:%(levelname)s:%(module)s:%(lineno)d:%(message)s")
#     worker = MPIteratorWorker(config, inbox_queue, outbox_queue, iteration_queue)
#
#     while True:
#         try:
#             worker.handle_incoming()
#         except Exception as e:
#             outbox_queue.put(e)
#             raise e
#
#
# class MPIterator:
#     def __init__(self, **config):
#         self.inbox_queue = Queue(maxsize=30)
#         self.outbox_queue = Queue()
#         self.iteration_queue = Queue(maxsize=30)
#
#         self.worker_proc = Process(
#             target=start_worker, args=(
#                 config,
#                 self.outbox_queue,
#                 self.inbox_queue,
#                 self.iteration_queue
#             )
#         )
#         self.worker_proc.start()
#         self.receive_init_confirmation()
#         self._stop_iteration = False
#
#     def receive_init_confirmation(self):
#         self.receive_expected(MPIteratorWorker.OutboxTypes.worker_started, timeout=600)
#
#     def receive_expected(self, expected_descriptor: Union[Enum, Set], timeout=None, queue=None):
#         keep_indefinitely = False
#         if timeout is None:
#             keep_indefinitely = True
#
#         if queue is None:
#             queue = self.inbox_queue
#         # logging.info(f"Receiving response {expected_descriptor}")
#
#         if keep_indefinitely:
#             # logging.info("Blocking until received")
#             while True:
#                 try:
#                     response: Union[Message, Exception] = queue.get(timeout=10)
#                     break
#                 except Empty:
#                     assert self.worker_proc.is_alive(), f"Worker in {self.__class__.__name__} is dead"
#                     # logging.info(f"Worker in {self.__class__.__name__} is still alive")
#         else:
#             # logging.info(f"Waiting {timeout} seconds to receive")
#             response: Union[Message, Exception] = queue.get(timeout=timeout)
#
#         if isinstance(response, Exception):
#             raise response
#
#         if not isinstance(expected_descriptor, set):
#             expected_descriptor = {expected_descriptor}
#         else:
#             pass
#
#         assert response.descriptor in expected_descriptor, f"Expected {expected_descriptor}, but received {response.descriptor}"
#         # logging.info(f"Received successfully")
#
#         return response.content
#
#     def send_request(self, request_descriptor, content=None):
#         # logging.info(f"Sending request {request_descriptor}")
#         self.outbox_queue.put(
#             Message(
#                 descriptor=request_descriptor,
#                 content=content
#             )
#         )
#
#     def send_request_and_receive_response(
#             self, request_descriptor, response_descriptor, content=None
#     ):
#         self.send_request(request_descriptor, content)
#         return self.receive_expected(response_descriptor)
#
#     def __iter__(self):
#         self.send_request(
#             request_descriptor=MPIteratorWorker.InboxTypes.iterate,
#             content=None
#         )
#
#         while True:
#             received = self.receive_expected(
#                 {MPIteratorWorker.OutboxTypes.next, MPIteratorWorker.OutboxTypes.stop_iteration},
#                 queue=self.iteration_queue
#             )
#             if received is None:
#                 break
#             yield received
#
#     def __len__(self):
#         self.send_request_and_receive_response(
#             MPIteratorWorker.InboxTypes.get_len, MPIteratorWorker.OutboxTypes.len
#         )


def test_batcher():
    import sys
    import json
    from transformers import RobertaTokenizer
    from SourceCodeTools.cli_arguments import default_graph_config
    data_path = sys.argv[1]
    cache_dir = sys.argv[2]

    decoder_mapping = RobertaTokenizer.from_pretrained("microsoft/codebert-base").decoder
    tok_ids, words = zip(*decoder_mapping.items())
    vocab_mapping = dict(zip(words, tok_ids))

    data = StreamIterator(data_path)

    batcher = HybridBatcher(
        # MPIterator(iter_fn=StreamIterator, path=data_path),
        data,
        batch_size=8, seq_len=512,
        wordmap=vocab_mapping, tagmap=None,
        class_weights=False, sort_by_length=False, tokenizer="codebert", no_localization=False,
        cache_dir=cache_dir, graph_config=default_graph_config(),
        graphmap=None, element_hash_size=1000
    )

    for batch in batcher:
        print()
        pass

    from tqdm import tqdm
    for line in tqdm(data):
        batcher.tokenize_and_encode(*line)


if __name__ == "__main__":
    test_batcher()