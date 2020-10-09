from typing import Iterable, List, Dict, Any, NamedTuple, Tuple, TypeVar, Callable
import numpy as np

Edge = Tuple[int, int]
Node = int
T = TypeVar("T")
Grid = Tuple[Tuple[T]]
State = Tuple[int]


IsingInstance = Dict[Tuple[Node, Node], float]


class Cluster:
    instance: IsingInstance
    nodes: Tuple[Node]
    legs: Dict[str, Dict[Node, Tuple[Node]]]  # {"up": {0: [5, 7], 10: [11, 14, 15], ...


class Tensor:
    data: np.ndarray


class PEPSTensor(Tensor):
    cluster: Cluster
    cluster_state: Tuple[State]


class PEPS:
    tensors: Grid[PEPSTensor]


class MPSTensor(Tensor):
    pass


class MPS:
    tensors: Tuple[MPSTensor]


graph_partition = Callable[[IsingInstance], Iterable[Cluster]]


def compute_stuff(instance: IsingInstance, partition_into_clusters: graph_partition):
    clusters = partition_into_clusters(instance)
    ...


def partition_chimera_into_grid(instance: IsingInstance) -> Iterable[Cluster]:
    pass


if __name__ == "__main__":
    pass
