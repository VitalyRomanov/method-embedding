from enum import Enum


class SGPartitionStrategies(Enum):
    package = "package"
    file = "file"
    mention = "mention"


class SGLabelSpec(Enum):
    nodes = "nodes"
    edges = "edges"
    subgraphs = "subgraphs"