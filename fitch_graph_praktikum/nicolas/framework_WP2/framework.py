import random

import networkx as nx
import copy
import numpy as np

from typing import TYPE_CHECKING, Optional, Callable, Literal

from fitch_graph_praktikum.nicolas.framework_WP2.random import partition_random

if TYPE_CHECKING:
    from typing import Any, Dict, Generator


class BasePartition:

    def __init__(self,
                 partition_list: list[list[frozenset]],
                 nodes: list[frozenset]):
        self.partition_list: list[list[frozenset]] = partition_list
        self.nodes: list[frozenset] = nodes

    def aggregate(self):
        self.partition_list = [
            [frozenset({node for entry in community for node in entry})] for community in self.partition_list
        ]
        self.nodes = [community[0] for community in self.partition_list]

    def get_nodes(self):
        return self.nodes

    def to_flat_partition(self):
        return [[node for entry in community for node in entry] for community in self.partition_list]

    def get_community(self, i):
        return self.partition_list[i]

    def get_community_for_base_node(self, base_node) -> int:
        for i, community in enumerate(self.partition_list):
            for entry in community:
                if base_node in entry:
                    return i
        raise KeyError(f'Base node {base_node} not in partition.')

    def __len__(self):
        return len(self.partition_list)

    def __str__(self):
        return self.to_flat_partition()


##################################
# Internal scoring functions #####
##################################


def log_sig(a, b):
    x0 = min(a, b)
    x1 = max(a, b)
    if x0 < 0.0:
        raise ValueError('function log_sig is only defined on positive values.')
    if x0 <= 0.0000000000000001 or x1 <= 0.0000000000000001:
        return 0.0
    return 2 / (1 + np.exp(-np.log2(x0 / x1)))


def _weight_per_edge_cut_data(graph: nx.Graph, partition: BasePartition):
    node_mapping = {node: None for node in graph.nodes}
    community_data = {i: {'weight': 0.0, 'edges': 0, 'nodes': 0} for i in range(len(partition))}
    external_edge_weights = 0.0

    min_weight = None
    max_weight = None

    num_edges = 0
    num_cut_edges = 0
    sum_weights = 0.0
    for node1, node2, data in graph.edges(data=True):
        num_edges += 1

        if min_weight is None or data['weight'] < min_weight:
            min_weight = data['weight']
        if max_weight is None or data['weight'] > max_weight:
            max_weight = data['weight']
        sum_weights += data['weight']

        if node_mapping[node1] is None:
            community1 = partition.get_community_for_base_node(node1)
            node_mapping[node1] = community1
            community_data[community1]['nodes'] += 1
        else:
            community1 = node_mapping[node1]
        if node_mapping[node2] is None:
            community2 = partition.get_community_for_base_node(node2)
            node_mapping[node2] = community2
            community_data[community2]['nodes'] += 1
        else:
            community2 = node_mapping[node2]

        if community1 != community2:
            external_edge_weights += data['weight']
            num_cut_edges += 1
        else:
            community_data[community1]['edges'] += 1
            community_data[community1]['weight'] += data['weight']

    return external_edge_weights, max_weight, sum_weights, num_cut_edges, num_edges, community_data


def average_weight_per_edge_cut(graph: nx.Graph, partition: BasePartition, val_lambda: float = 1) -> "float":
    external_edge_weights, max_weight, _, num_cut_edges, num_edges, community_data = \
        _weight_per_edge_cut_data(graph=graph, partition=partition)

    # Internal term
    internal_degrees = 0
    for i, data in community_data.items():
        # squared sum of internal degrees
        internal_degrees += (2 * data['edges']) * (2 * data['edges'])
    internal_term = 0.01 * internal_degrees / (2 * num_edges)  # rescale degree

    # External term
    average_weight_per_cut_edge = external_edge_weights / num_cut_edges if num_cut_edges > 0 else 0.0
    external_term = log_sig(average_weight_per_cut_edge, max_weight)

    return external_term + val_lambda * internal_term


def average_weight_per_edge_cut2(graph: nx.Graph, partition: BasePartition) -> "float":
    external_edge_weights, max_weight, _, num_cut_edges, _, _ = \
        _weight_per_edge_cut_data(graph=graph, partition=partition)

    # External term
    average_weight_per_cut_edge = external_edge_weights / num_cut_edges if num_cut_edges > 0 else 0.0
    external_term = log_sig(average_weight_per_cut_edge, max_weight)

    return external_term


def sum_weight_per_edge_cut(graph: nx.Graph, partition: BasePartition) -> "float":
    external_edge_weights, _, sum_weights, num_cut_edges, _, _ = \
        _weight_per_edge_cut_data(graph=graph, partition=partition)

    # External term
    average_weight_per_cut_edge = external_edge_weights / num_cut_edges if num_cut_edges > 0 else 0.0
    external_term = log_sig(average_weight_per_cut_edge, sum_weights)

    return external_term


################################
# Post-Processing function #####
################################


def convert_to_bi_partition(
        graph: nx.Graph,
        k_partition: list[list],
        mod_func: Callable[[list, list, dict[tuple[int, int]]], float]) -> "list[list]":
    if len(k_partition) == 1:
        return partition_random(graph)
    if len(k_partition) == 2:
        return k_partition

    relations = {(x1, x2): v['weight'] for x1, x2, v in graph.edges(data=True)}

    mapping = {}
    for i in range(len(k_partition) - 1):
        for j in range(i, len(k_partition)):
            val = mod_func(k_partition[i], k_partition[j], relations)
            mapping[(i, j)] = val
            mapping[(j, i)] = val

    max_edge = max(mapping.keys(), key=mapping.get)

    remaining = [i for i in range(len(k_partition)) if i not in max_edge]
    new_partition = [[max_edge[0]], [max_edge[1]]]
    while remaining:
        # find edge with minimum weight
        min_node = None
        min_weight = None
        min_dir = 0
        for i in remaining:
            min_left = min(mapping[(i, j)] for j in range(len(new_partition[0])))
            min_right = min(mapping[(i, j)] for j in range(len(new_partition[1])))

            new_dir = 0
            new_min = min_left
            if min_right <= min_left:
                new_dir = 1
                new_min = min_right

            if min_weight is None or new_min < min_weight:
                min_weight = new_min
                min_dir = new_dir
                min_node = i

        remaining.remove(min_node)
        new_partition[min_dir].append(min_node)

    return [
        list(sorted(node for i in new_partition[0] for node in k_partition[i])),
        list(sorted(node for i in new_partition[1] for node in k_partition[i]))
    ]
