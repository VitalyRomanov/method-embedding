import logging
from functools import lru_cache
from multiprocessing import Queue, Process
from os.path import isfile
from enum import Enum
from queue import Empty
from time import sleep
from typing import Union, Set

from SourceCodeTools.code.data.GraphStorage import AbstractGraphStorage, OnDiskGraphStorageWithFastIteration


class Message:
    def __init__(self, descriptor, content):
        self.descriptor = descriptor
        self.content = content


class GraphStorageWorker:
    class InboxTypes(Enum):
        iterate_subgraphs = 0
        get_node_type_descriptions = 1
        get_edge_type_descriptions = 2
        get_nodes_with_subwords = 3
        get_nodes = 4
        get_edges = 5
        get_node_types = 6
        get_nodes_for_classification = 7
        get_info_for_node_ids = 8
        get_num_nodes = 10
        get_num_edges = 11
        get_info_for_subgraphs = 12
        get_info_for_edge_ids = 13

    class OutboxTypes(Enum):
        iterate_subgraphs = 0
        node_type_descriptions = 1
        edge_type_descriptions = 2
        nodes_with_subwords = 3
        nodes = 4
        edges = 5
        node_types = 6
        nodes_for_classification = 7
        info_for_node_ids = 8
        stop_iteration = 9
        num_nodes = 10
        num_edges = 11
        info_for_subgraphs = 12
        info_for_edge_ids = 13
        worker_started = 14

    inbox_queue: Queue
    outbox_queue: Queue
    iteration_queue: Queue
    dataset_db: AbstractGraphStorage
    # _stop_iteration = False

    def __init__(self, config, inbox_queue, outbox_queue, iteration_queue):
        storage_class = config["storage_class"]
        database_exists = isfile(config["db_path"])

        self.dataset_db = storage_class(config["db_path"])
        if not database_exists:
            self.dataset_db.import_from_files(config["data_path"])
        #     self.dataset_db.import_from_files(self.data_path)
        # self.dataset_db = OnDiskGraphStorage(config["path"])
        self.inbox_queue = inbox_queue
        self.outbox_queue = outbox_queue
        self.iteration_queue = iteration_queue
        self._send_init_confirmation()

    def _send_init_confirmation(self):
        self.send_out(Message(
            descriptor=GraphStorageWorker.OutboxTypes.worker_started,
            content=None
        ))

    def send_out(self, message, queue=None, keep_trying=True, verbose=True) -> True:
        # if verbose:
        #     logging.info(f"Attempting to send {message.descriptor}")
        if queue is None:
            queue = self.outbox_queue

        if keep_trying:
            # logging.info(f"Blocking until sent")
            while queue.full():
                sleep(0.2)
        else:
            if queue.full():
                return False
        # logging.info(f"There is a spot in the queue")

        queue.put(message)
        # logging.info(f"Message sent")
        return True

    def check_for_new_messages(self, verbose=True):
        interrupt_iteration = False
        try:
            # if verbose:
            #     logging.info("Checking incoming requests")
            message = self.inbox_queue.get(timeout=0.2)
            if message.descriptor == self.InboxTypes.iterate_subgraphs:
                # logging.info("New iteration request has arrived, begin cleanup")
                while not self.outbox_queue.empty():
                    self.outbox_queue.get()
                # logging.info("Emptied iteration queue")
                self.inbox_queue.put(message)
                # logging.info("Putting iteration request back")
                assert self.inbox_queue.empty(), "Trying to put iteration request back, but there is already something else"
                interrupt_iteration = True
            else:
                # logging.info("Found a new request, handling")
                self._handle_message(message)
        except Empty:
            pass
        return interrupt_iteration

    def _iterate_subgraphs(self, **kwargs):
        for subgraph in self.dataset_db.iterate_subgraphs(**kwargs):
            sent = False
            attempts = 0
            while not sent:
                verbose = attempts % 100 == 0
                interrupt_iteration = self.check_for_new_messages(verbose=verbose)
                # if verbose:
                #     logging.info("Trying to send subgraph")
                if interrupt_iteration:
                    return
                sent = self.send_out(Message(
                    descriptor=GraphStorageWorker.OutboxTypes.iterate_subgraphs,
                    content=subgraph
                ), queue=self.iteration_queue, keep_trying=False, verbose=verbose)
                attempts += 1
        self.send_out(Message(
            descriptor=GraphStorageWorker.OutboxTypes.stop_iteration,
            content=None
        ), queue=self.iteration_queue)

    def _handle_message(self, message):
        kwargs = message.content
        if kwargs is None:
            kwargs = {}

        if message.descriptor == GraphStorageWorker.InboxTypes.iterate_subgraphs:
            self._iterate_subgraphs(**kwargs)

        elif message.descriptor == GraphStorageWorker.InboxTypes.get_node_type_descriptions:
            self.send_out(Message(
                descriptor=GraphStorageWorker.OutboxTypes.node_type_descriptions,
                content=self.dataset_db.get_node_type_descriptions()
            ))

        elif message.descriptor == GraphStorageWorker.InboxTypes.get_edge_type_descriptions:
            self.send_out(Message(
                descriptor=GraphStorageWorker.OutboxTypes.edge_type_descriptions,
                content=self.dataset_db.get_edge_type_descriptions()
            ))
        elif message.descriptor == GraphStorageWorker.InboxTypes.get_nodes_with_subwords:
            self.send_out(Message(
                descriptor=GraphStorageWorker.OutboxTypes.nodes_with_subwords,
                content=self.dataset_db.get_nodes_with_subwords()
            ))

        elif message.descriptor == GraphStorageWorker.InboxTypes.get_nodes:
            self.send_out(Message(
                descriptor=GraphStorageWorker.OutboxTypes.nodes,
                content=self.dataset_db.get_nodes(**kwargs)
            ))

        elif message.descriptor == GraphStorageWorker.InboxTypes.get_edges:
            self.send_out(Message(
                descriptor=GraphStorageWorker.OutboxTypes.edges,
                content=self.dataset_db.get_edges(**kwargs)
            ))

        elif message.descriptor == GraphStorageWorker.InboxTypes.get_node_types:
            self.send_out(Message(
                descriptor=GraphStorageWorker.OutboxTypes.node_types,
                content=self.dataset_db.get_node_types()
            ))

        elif message.descriptor == GraphStorageWorker.InboxTypes.get_nodes_for_classification:
            self.send_out(Message(
                descriptor=GraphStorageWorker.OutboxTypes.nodes_for_classification,
                content=self.dataset_db.get_nodes_for_classification()
            ))

        elif message.descriptor == GraphStorageWorker.InboxTypes.get_info_for_node_ids:
            self.send_out(Message(
                descriptor=GraphStorageWorker.OutboxTypes.info_for_node_ids,
                content=self.dataset_db.get_info_for_node_ids(**kwargs)
            ))

        # elif message.descriptor == GraphStorageWorker.InboxTypes.stop_iteration:
        #     # self.send_out(Message(
        #     #     descriptor=GraphStorageWorker.OutboxTypes.info_for_node_ids,
        #     #     content=self.dataset_db.get_info_for_node_ids(**kwargs)
        #     # ))
        #     raise NotImplementedError

        elif message.descriptor == GraphStorageWorker.InboxTypes.get_num_nodes:
            self.send_out(Message(
                descriptor=GraphStorageWorker.OutboxTypes.num_nodes,
                content=self.dataset_db.get_num_nodes(**kwargs)
            ))

        elif message.descriptor == GraphStorageWorker.InboxTypes.get_num_edges:
            self.send_out(Message(
                descriptor=GraphStorageWorker.OutboxTypes.num_edges,
                content=self.dataset_db.get_num_edges(**kwargs)
            ))

        elif message.descriptor == GraphStorageWorker.InboxTypes.get_info_for_subgraphs:
            self.send_out(Message(
                descriptor=GraphStorageWorker.OutboxTypes.info_for_subgraphs,
                content=self.dataset_db.get_info_for_subgraphs(**kwargs)
            ))

        elif message.descriptor == GraphStorageWorker.InboxTypes.get_info_for_edge_ids:
            self.send_out(Message(
                descriptor=GraphStorageWorker.OutboxTypes.info_for_edge_ids,
                content=self.dataset_db.get_info_for_edge_ids(**kwargs)
            ))

        else:
            raise ValueError(f"Unrecognized message descriptor: {message.descriptor.name}")

    def handle_incoming(self):
        message = self.inbox_queue.get()
        response = self._handle_message(message)
        # if response is not None:
        #     self.outbox_queue.put(response)


def start_worker(config, inbox_queue, outbox_queue, iteration_queue, *args, **kwargs):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(module)s:%(lineno)d:%(message)s")
    worker = GraphStorageWorker(config, inbox_queue, outbox_queue, iteration_queue)

    while True:
        try:
            worker.handle_incoming()
        except Exception as e:
            outbox_queue.put(e)
            raise e


class GraphStorageWorkerAdapter(AbstractGraphStorage):
    def __init__(self, path, storage_class):
        self.inbox_queue = Queue(maxsize=30)
        self.outbox_queue = Queue()
        self.iteration_queue = Queue(maxsize=30)

        self.worker_proc = Process(
            target=start_worker, args=(
                {"db_path": path, "storage_class": storage_class},
                self.outbox_queue,
                self.inbox_queue,
                self.iteration_queue
            )
        )
        # self.history = []
        self.worker_proc.start()
        self.receive_init_confirmation()
        # self._stop_iteration = False

    # def stop_iteration(self):
    #     self._stop_iteration = True

    def receive_init_confirmation(self):
        self.receive_expected(GraphStorageWorker.OutboxTypes.worker_started, timeout=10)

    def receive_expected(self, expected_descriptor: Union[Enum, Set], timeout=None, queue=None):
        keep_indefinitely = False
        if timeout is None:
            keep_indefinitely = True

        if queue is None:
            queue = self.inbox_queue
        # logging.info(f"Receiving response {expected_descriptor}")

        if keep_indefinitely:
            # logging.info("Blocking until received")
            while True:
                try:
                    response: Union[Message, Exception] = queue.get(timeout=10)
                    break
                except Empty:
                    assert self.worker_proc.is_alive(), f"Worker in {self.__class__.__name__} is dead"
                    # logging.info(f"Worker in {self.__class__.__name__} is still alive")
        else:
            # logging.info(f"Waiting {timeout} seconds to receive")
            response: Union[Message, Exception] = queue.get(timeout=timeout)

        if isinstance(response, Exception):
            raise response

        if not isinstance(expected_descriptor, set):
            expected_descriptor = {expected_descriptor}
        else:
            pass

        assert response.descriptor in expected_descriptor, f"Expected {expected_descriptor}, but received {response.descriptor}"
        # logging.info(f"Received successfully {response.descriptor}")
        return response.content

    def send_request(self, request_descriptor, content=None):
        # logging.info(f"Sending request {request_descriptor}")
        self.outbox_queue.put(
            Message(
                descriptor=request_descriptor,
                content=content
            )
        )

    def send_request_and_receive_response(
            self, request_descriptor, response_descriptor, content=None
    ):
        self.send_request(request_descriptor, content)
        return self.receive_expected(response_descriptor)

    @lru_cache
    def get_node_type_descriptions(self):
        return self.send_request_and_receive_response(
            request_descriptor=GraphStorageWorker.InboxTypes.get_node_type_descriptions,
            response_descriptor=GraphStorageWorker.OutboxTypes.node_type_descriptions
        )

    @lru_cache
    def get_edge_type_descriptions(self):
        return self.send_request_and_receive_response(
            request_descriptor=GraphStorageWorker.InboxTypes.get_edge_type_descriptions,
            response_descriptor=GraphStorageWorker.OutboxTypes.edge_type_descriptions
        )

    def get_num_nodes(self):
        return self.send_request_and_receive_response(
            request_descriptor=GraphStorageWorker.InboxTypes.get_num_nodes,
            response_descriptor=GraphStorageWorker.OutboxTypes.num_nodes
        )

    def get_num_edges(self):
        return self.send_request_and_receive_response(
            request_descriptor=GraphStorageWorker.InboxTypes.get_num_edges,
            response_descriptor=GraphStorageWorker.OutboxTypes.num_edges
        )

    def get_nodes(self, type_filter=None):
        return self.send_request_and_receive_response(
            request_descriptor=GraphStorageWorker.InboxTypes.get_nodes,
            response_descriptor=GraphStorageWorker.OutboxTypes.nodes,
            content={
                "type_filter": type_filter
            }
        )

    def get_edges(self, type_filter=None):
        return self.send_request_and_receive_response(
            request_descriptor=GraphStorageWorker.InboxTypes.get_edges,
            response_descriptor=GraphStorageWorker.OutboxTypes.edges,
            content={
                "type_filter": type_filter
            }
        )

    def iterate_subgraphs(self, how, groups):
        self.send_request(
            GraphStorageWorker.InboxTypes.iterate_subgraphs,
            content={
                "how": how,
                "groups": groups
            }
        )

        while True:
            # if self._stop_iteration:
            #     self._stop_iteration = False
            #     break
            received = self.receive_expected(
                {GraphStorageWorker.OutboxTypes.iterate_subgraphs, GraphStorageWorker.OutboxTypes.stop_iteration},
                queue=self.iteration_queue
            )
            if received is None:
                break
            yield received

    def get_nodes_with_subwords(self):
        return self.send_request_and_receive_response(
            request_descriptor=GraphStorageWorker.InboxTypes.get_nodes_with_subwords,
            response_descriptor=GraphStorageWorker.OutboxTypes.nodes_with_subwords,
        )

    def get_nodes_for_classification(self):
        return self.send_request_and_receive_response(
            request_descriptor=GraphStorageWorker.InboxTypes.get_nodes_for_classification,
            response_descriptor=GraphStorageWorker.OutboxTypes.nodes_for_classification,
        )

    def get_info_for_node_ids(self, node_ids, field):
        return self.send_request_and_receive_response(
            request_descriptor=GraphStorageWorker.InboxTypes.get_info_for_node_ids,
            response_descriptor=GraphStorageWorker.OutboxTypes.info_for_node_ids,
            content={
                "node_ids": node_ids,
                "field": field
            }
        )

    def get_info_for_subgraphs(self, subgraph_ids, field):
        return self.send_request_and_receive_response(
            request_descriptor=GraphStorageWorker.InboxTypes.get_info_for_subgraphs,
            response_descriptor=GraphStorageWorker.OutboxTypes.info_for_subgraphs,
            content={
                "subgraph_ids": subgraph_ids,
                "field": field
            }
        )

    def get_info_for_edge_ids(self, edge_ids, field):
        return self.send_request_and_receive_response(
            request_descriptor=GraphStorageWorker.InboxTypes.get_info_for_edge_ids,
            response_descriptor=GraphStorageWorker.OutboxTypes.info_for_edge_ids,
            content={
                "edge_ids": edge_ids,
                "field": field
            }
        )

    def __del__(self):
        # stop iteration
        # terminate
        ...


def test_adapter():
    storage = GraphStorageWorkerAdapter(
        path="/Users/LTV/dev/method-embeddings/examples/large_graph/dataset.db",
        storage_class=OnDiskGraphStorageWithFastIteration
    )

    from SourceCodeTools.code.data.dataset.partition_strategies import SGPartitionStrategies
    iterator = storage.iterate_subgraphs(SGPartitionStrategies.mention, groups=None)

    for i in range(4):
        next(iterator)

    print()


if __name__ == "__main__":
    test_adapter()
