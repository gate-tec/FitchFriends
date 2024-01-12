import random

import networkx as nx
import copy
import numpy as np

from typing import TYPE_CHECKING, Optional, Callable, Literal

from fitch_graph_praktikum.alex.WP2.weight_scoring_for_partitioning import average_weight_scoring, sum_weight_scoring
from fitch_graph_praktikum.nicolas.framework_WP2.framework import convert_to_bi_partition
from fitch_graph_praktikum.nicolas.framework_WP2.random import partition_random
from fitch_graph_praktikum.nicolas.framework_WP2.leiden import partition_leiden, LeidenLoopError
from fitch_graph_praktikum.nicolas.framework_WP2.louvain import partition_louvain

if TYPE_CHECKING:
    from typing import Any, Dict, Generator


if __name__ == '__main__':
    test_graph = nx.Graph()
    test_graph.add_edge(0, 1, weight=random.random())
    test_graph.add_edge(1, 2, weight=random.random())
    test_graph.add_edge(2, 0, weight=random.random())
    test_graph.add_edge(0, 3, weight=random.random())
    test_graph.add_edge(3, 4, weight=random.random())

    print(test_graph.edges(data=True))

    test_part = partition_louvain(test_graph)
    print(test_part)

    try:
        test_part = partition_leiden(test_graph, val_gamma=1.0, theta=0.01)
        print(test_part)
    except LeidenLoopError as err:
        print("Loop:")
        print(err.last_common_partition.to_flat_partition())

    print(convert_to_bi_partition(
        graph=test_graph,
        k_partition=[[0, 1], [2], [3], [4]],
        mod_func=average_weight_scoring
    ))
