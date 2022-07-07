from enum import Enum
from multiprocessing import Queue, Process
from queue import Empty  # , Queue
# from threading import Thread

from dgl.dataloading import NodeDataLoader


class Message:
    def __init__(self, descriptor, content):
        self.descriptor = descriptor
        self.content = content


class NewProcessNodeDataLoader:
    def __init__(self, graph, nodes_for_batching, sampler, batch_size):
        self.graph = graph
        self.sampler = sampler
        self.nodes_for_batching = nodes_for_batching
        self.batch_size = batch_size

        self.process = None
        self.inbox_queue = None
        self.outbox_queue = None

    def start_process(self):
        self.outbox_queue = Queue()
        self.inbox_queue = Queue()

        self.process = Process(
            target=start_worker, args=(
                {
                    "graph": self.graph,
                    "sampler": self.sampler,
                    "nodes_for_batching": self.nodes_for_batching,
                    "batch_size": self.batch_size
                },
                self.outbox_queue,
                self.inbox_queue
            )
        )
        self.process.start()
        self.ensure_process_started()

    def ensure_process_started(self):
        message = self.inbox_queue.get()
        assert message.descriptor == NewProcessNodeDataLoaderProducer.OutboxTypes.process_ready

    def terminate_existing(self):
        self.outbox_queue.put(Message(
            descriptor=NewProcessNodeDataLoaderProducer.InboxTypes.terminate,
            content=None
        ))
        self.outbox_queue.close()
        self.inbox_queue.close()
        self.process.join()
        self.process.terminate()
        self.process.close()

    def terminate(self):
        self.terminate_existing()

    def __iter__(self):

        if self.process is not None:
            self.terminate_existing()

        self.start_process()
        self.outbox_queue.put(Message(
            descriptor=NewProcessNodeDataLoaderProducer.InboxTypes.start_iteration,
            content=None
        ))

        return self

    def __next__(self):
        message = self.inbox_queue.get()
        # assert message.descriptor == NewProcessNodeDataLoaderProducer.OutboxTypes.batch
        if message.descriptor == NewProcessNodeDataLoaderProducer.OutboxTypes.batch:
            content = message.content
            return content["input_nodes"], content["seeds"], content["blocks"]
        elif message.descriptor == NewProcessNodeDataLoaderProducer.OutboxTypes.stop_iteration:
            raise StopIteration()
        else:
            raise Exception()


class NewProcessNodeDataLoaderProducer:
    class InboxTypes(Enum):
        start_iteration = 0
        terminate = 1

    class OutboxTypes(Enum):
        batch = 0
        stop_iteration = 1
        process_ready = 2

    def __init__(self, config, inbox_queue, outbox_queue):
        self.graph = config["graph"]
        self.sampler = config["sampler"]
        self.nodes_for_batching = config["nodes_for_batching"]
        self.batch_size = config["batch_size"]
        self.terminate = False

        self.loader = NodeDataLoader(
            self.graph, self.nodes_for_batching, self.sampler, batch_size=self.batch_size, shuffle=True
        )
        self.inbox_queue = inbox_queue
        self.outbox_queue = outbox_queue
        self.outbox_queue.put(Message(
            descriptor=NewProcessNodeDataLoaderProducer.OutboxTypes.process_ready,
            content=None
        ))

    def get_message(self, timeout=None):
        try:
            return self.inbox_queue.get(timeout=timeout)
        except ValueError:
            self.terminate = True
        return None

    def send_message(self, message):
        try:
            self.outbox_queue.put(message)
        except ValueError:
            self.terminate = True
        return None

    def _iterate_subgraphs(self, **kwargs):
        for input_nodes, seeds, blocks in self.loader:
            self.outbox_queue.put(Message(
                descriptor=NewProcessNodeDataLoaderProducer.OutboxTypes.batch,
                content={
                    "input_nodes": input_nodes,
                    "seeds": seeds,
                    "blocks": blocks
                }
            ))
        self.outbox_queue.put(Message(
            descriptor=NewProcessNodeDataLoaderProducer.OutboxTypes.stop_iteration,
            content=None
        ))

    def _handle_message(self, message):
        kwargs = message.content
        if message.descriptor == NewProcessNodeDataLoaderProducer.InboxTypes.start_iteration:
            self._iterate_subgraphs()
        elif message.descriptor == NewProcessNodeDataLoaderProducer.InboxTypes.terminate:
            self.terminate = True
        else:
            raise ValueError("Unrecognized message descriptor")

    def handle_incoming(self):
        try:
            message = self.inbox_queue.get(timeout=3)
            response = self._handle_message(message)

            if response is not None:
                self.outbox_queue.put(response)
        except Empty:
            pass

        return self.terminate


def start_worker(config, inbox_queue, outbox_queue, *args, **kwargs):
    worker = NewProcessNodeDataLoaderProducer(config, inbox_queue, outbox_queue)

    while True:
        terminate = worker.handle_incoming()
        if terminate:
            break
