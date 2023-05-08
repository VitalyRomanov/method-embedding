import logging
from copy import copy
from torch.multiprocessing import Queue, Process
from enum import Enum
from queue import Empty
from time import sleep
from typing import Union, Set

from SourceCodeTools.code.data.dataset.DataLoader import SGAbstractDataLoader
from SourceCodeTools.code.data.dataset.Dataset import SourceGraphDataset


class Message:
    def __init__(self, descriptor, content):
        self.descriptor = descriptor
        self.content = content


class DataLoaderWorker:
    class InboxTypes(Enum):
        get_label_loader = 0
        get_label_encoder = 2
        get_train_num_batches = 3
        get_test_num_batches = 4
        get_val_num_batches = 5
        get_any_num_batches = 6
        iterate_subgraphs = 7
        get_num_classes = 8

    class OutboxTypes(Enum):
        label_loader = 0
        label_encoder = 2
        train_num_batches = 3
        test_num_batches = 4
        val_num_batches = 5
        any_num_batches = 6
        iterate_subgraphs = 7
        stop_iteration = 8
        worker_started = 9
        num_classes = 10

    inbox_queue: Queue
    outbox_queue: Queue
    iteration_queue: Queue
    dataloader: SGAbstractDataLoader
    _stop_iteration = False

    def __init__(self, config, dataloader_class, inbox_queue, outbox_queue, iteration_queue):
        dataset_config = config["dataset"]
        config["dataset"] = SourceGraphDataset(**dataset_config)
        self.dataloader = dataloader_class(**config)

        self.inbox_queue = inbox_queue
        self.outbox_queue = outbox_queue
        self.iteration_queue = iteration_queue
        self._send_init_confirmation()

    def _send_init_confirmation(self):
        self.send_out(Message(
            descriptor=DataLoaderWorker.OutboxTypes.worker_started,
            content=None
        ))

    def send_out(self, message, queue=None, keep_trying=True) -> True:
        """
        :param message:
        :param queue:
        :param keep_trying: Block until can put the item in the queue
        :return: Return True if could put item in the queue, and False otherwise
        """
        if queue is None:
            queue = self.outbox_queue

        if keep_trying:
            while queue.full():
                sleep(0.2)
        else:
            if queue.full():
                return False

        queue.put(message)
        return True

    def check_for_new_messages(self):
        interrupt_iteration = False
        try:
            message = self.inbox_queue.get(timeout=0.2)
            if message.descriptor == self.InboxTypes.iterate_subgraphs:
                while not self.outbox_queue.empty():
                    self.outbox_queue.get()
                self.inbox_queue.put(message)
                interrupt_iteration = True
            else:
                self._handle_message(message)
        except Empty:
            pass
        return interrupt_iteration

    def _iterate_subgraphs(self, partition_label):
        partition_iterator = self.dataloader.partition_iterator(partition_label)

        for subgraph in partition_iterator:
            sent = False
            subgraph_ = copy(subgraph)
            subgraph_["blocks"] = copy(subgraph["blocks"])
            while not sent:
                interrupt_iteration = self.check_for_new_messages()
                if interrupt_iteration:
                    return
                sent = self.send_out(Message(
                    descriptor=DataLoaderWorker.OutboxTypes.iterate_subgraphs,
                    content=subgraph_
                ), queue=self.iteration_queue, keep_trying=False)
        self.send_out(Message(
            descriptor=DataLoaderWorker.OutboxTypes.stop_iteration,
            content=None
        ), queue=self.iteration_queue)

    def _handle_message(self, message):
        if message.descriptor == DataLoaderWorker.InboxTypes.iterate_subgraphs:
            self._iterate_subgraphs(message.content)

        elif message.descriptor == DataLoaderWorker.InboxTypes.get_label_loader:
            self.send_out(Message(
                descriptor=DataLoaderWorker.OutboxTypes.label_loader,
                content=self.dataloader.get_label_loader(message.content)
            ))

        elif message.descriptor == DataLoaderWorker.InboxTypes.get_label_encoder:
            self.send_out(Message(
                descriptor=DataLoaderWorker.OutboxTypes.label_encoder,
                content=self.dataloader.get_label_encoder()
            ))
        elif message.descriptor == DataLoaderWorker.InboxTypes.get_train_num_batches:
            self.send_out(Message(
                descriptor=DataLoaderWorker.OutboxTypes.train_num_batches,
                content=self.dataloader.get_train_num_batches()
            ))

        elif message.descriptor == DataLoaderWorker.InboxTypes.get_test_num_batches:
            self.send_out(Message(
                descriptor=DataLoaderWorker.OutboxTypes.test_num_batches,
                content=self.dataloader.get_test_num_batches()
            ))

        elif message.descriptor == DataLoaderWorker.InboxTypes.get_val_num_batches:
            self.send_out(Message(
                descriptor=DataLoaderWorker.OutboxTypes.val_num_batches,
                content=self.dataloader.get_val_num_batches()
            ))

        elif message.descriptor == DataLoaderWorker.InboxTypes.get_any_num_batches:
            self.send_out(Message(
                descriptor=DataLoaderWorker.OutboxTypes.any_num_batches,
                content=self.dataloader.get_any_num_batches()
            ))

        elif message.descriptor == DataLoaderWorker.InboxTypes.get_num_classes:
            self.send_out(Message(
                descriptor=DataLoaderWorker.OutboxTypes.num_classes,
                content=self.dataloader.get_num_classes()
            ))

        else:
            raise ValueError(f"Unrecognized message descriptor: {message.descriptor.name}")

    def handle_incoming(self):
        message = self.inbox_queue.get()
        response = self._handle_message(message)
        # if response is not None:
        #     self.outbox_queue.put(response)


def start_worker(config, dataloader_class, inbox_queue, outbox_queue, iteration_queue, *args, **kwargs):
    # import SourceCodeTools.code.data.dataset.DataLoader as dl
    # dataloader_class = getattr(dl, f"{dataloader_class}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(module)s:%(lineno)d:%(message)s")
    worker = DataLoaderWorker(config, dataloader_class, inbox_queue, outbox_queue, iteration_queue)

    while True:
        try:
            worker.handle_incoming()
        except Exception as e:
            outbox_queue.put(e)
            raise e


class DataLoaderWorkerAdapter(SGAbstractDataLoader):
    inbox_queue: Queue
    outbox_queue: Queue
    iteration_queue: Queue

    def __init__(self, dataloader_class=None, **config):
        super(DataLoaderWorkerAdapter, self).__init__(**config)
        self.inbox_queue = Queue(maxsize=30)
        self.outbox_queue = Queue()
        self.iteration_queue = Queue(maxsize=30)

        assert dataloader_class is not None

        self.worker_proc = Process(
            target=start_worker, args=(
                config,
                dataloader_class,
                self.outbox_queue,
                self.inbox_queue,
                self.iteration_queue
            )
        )
        # self.history = []
        self.worker_proc.start()
        self.receive_init_confirmation()
        self._stop_iteration = False

    def _initialize(
            self, dataset, labels_for, number_of_hops, batch_size, preload_for="package", labels=None,
            masker_fn=None, label_loader_class=None, label_loader_params=None, device="cpu",
            negative_sampling_strategy="w2v", neg_sampling_factor=1, base_path=None, objective_name=None,
            embedding_table_size=300000
    ):
        pass

    def get_label_loader(self, partition):
        return self.send_request_and_receive_response(
            request_descriptor=DataLoaderWorker.InboxTypes.get_label_loader,
            response_descriptor=DataLoaderWorker.OutboxTypes.label_loader,
            content=partition
        )

    def get_label_encoder(self):
        return self.send_request_and_receive_response(
            request_descriptor=DataLoaderWorker.InboxTypes.get_label_encoder,
            response_descriptor=DataLoaderWorker.OutboxTypes.label_encoder,
        )

    def get_train_num_batches(self):
        return self.send_request_and_receive_response(
            request_descriptor=DataLoaderWorker.InboxTypes.get_train_num_batches,
            response_descriptor=DataLoaderWorker.OutboxTypes.train_num_batches,
        )

    def get_test_num_batches(self):
        return self.send_request_and_receive_response(
            request_descriptor=DataLoaderWorker.InboxTypes.get_test_num_batches,
            response_descriptor=DataLoaderWorker.OutboxTypes.test_num_batches,
        )

    def get_val_num_batches(self):
        return self.send_request_and_receive_response(
            request_descriptor=DataLoaderWorker.InboxTypes.get_val_num_batches,
            response_descriptor=DataLoaderWorker.OutboxTypes.val_num_batches,
        )

    def get_any_num_batches(self):
        return self.send_request_and_receive_response(
            request_descriptor=DataLoaderWorker.InboxTypes.get_any_num_batches,
            response_descriptor=DataLoaderWorker.OutboxTypes.any_num_batches,
        )

    def get_num_classes(self):
        return self.send_request_and_receive_response(
            request_descriptor=DataLoaderWorker.InboxTypes.get_num_classes,
            response_descriptor=DataLoaderWorker.OutboxTypes.num_classes,
        )

    def receive_init_confirmation(self):
        self.receive_expected(DataLoaderWorker.OutboxTypes.worker_started, timeout=600)

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
        # logging.info(f"Received successfully")

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

    def iterate_partition(self, partition_label):
        self.send_request(
            request_descriptor=DataLoaderWorker.InboxTypes.iterate_subgraphs,
            content=partition_label
        )

        while True:
            received = self.receive_expected(
                {DataLoaderWorker.OutboxTypes.iterate_subgraphs, DataLoaderWorker.OutboxTypes.stop_iteration},
                queue=self.iteration_queue
            )
            if received is None:
                break
            yield received

    def partition_iterator(self, partition_label):
        iterate_partition = self.iterate_partition

        class MPSGDLIter:
            def __init__(self):
                self.partition_label = partition_label

            def __iter__(self):
                return iterate_partition(self.partition_label)

        return MPSGDLIter()
