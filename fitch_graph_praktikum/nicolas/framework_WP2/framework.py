import random

import networkx as nx
import copy
import numpy as np

from typing import TYPE_CHECKING, Optional, Callable, Literal

if TYPE_CHECKING:
    from typing import Any, Dict, Generator


def log_sig(a, b):
    x0 = min(a, b)
    x1 = max(a, b)
    if x0 < 0.0:
        raise ValueError('function log_sig is only defined on positive values.')
    if x0 <= 0.0000000000000001 or x1 <= 0.0000000000000001:
        return 0.0
    return 2 / (1 + np.exp(-np.log2(x0 / x1)))


class Partition:

    def __init__(self,
                 partition_list: list[list[frozenset]],
                 nodes: list[frozenset],
                 neighbors: dict[frozenset, dict[frozenset, dict[Literal['weight'], float]]]):
        self.partition_list: list[list[frozenset]] = partition_list
        self.nodes: list[frozenset] = nodes
        self.neighbors = neighbors

    @classmethod
    def singleton_partition_from_graph(cls, graph: nx.Graph) -> "Partition":
        """Initialize singleton partition from a list of nodes."""
        # convert all node representations to frozensets
        relabeled_graph = nx.relabel_nodes(graph, {node: frozenset({node}) for node in graph.nodes})
        # get relations and weights from base graph
        neighbors = {node: set(relabeled_graph[node].keys()) for node in relabeled_graph.nodes}
        # construct singleton partition
        return cls([[node] for node in relabeled_graph.nodes], [node for node in relabeled_graph.nodes], neighbors)

    def get_partitions_for_node(self, node) -> "Generator[Partition]":
        # find community of node
        community_index = -1
        for i in range(len(self.partition_list)):
            if node in self.partition_list[i]:
                community_index = i
                break
        # sanity check
        if community_index == -1:
            raise ValueError("Corresponding community not found")

        partial_partition_list = copy.deepcopy(self.partition_list)
        partial_partition_list[community_index].remove(node)

        if len(partial_partition_list[community_index]) > 0:
            # moving node in its own community yields a different partition
            new_partition_list = copy.deepcopy(partial_partition_list)
            new_partition_list.append([node])
            yield Partition(new_partition_list, self.nodes, self.neighbors)
        else:
            # remove empty community
            del partial_partition_list[community_index]
            community_index = -1

        for i in range(len(partial_partition_list)):
            if i == community_index:
                # skip old community
                continue
            new_partition_list = copy.deepcopy(partial_partition_list)
            new_partition_list[i].append(node)
            yield Partition(new_partition_list, self.nodes, self.neighbors)

    def aggregate(self):
        new_partition_list = [
            [frozenset({node for entry in community for node in entry})] for community in self.partition_list
        ]
        self.nodes = [community[0] for community in new_partition_list]
        # recalculate neighbors (using average weight per aggregated edges)
        # TODO: only necessary for Leiden (possibly class extension)
        new_neighbors = {node: set() for node in self.nodes}
        for i, old_community in enumerate(self.partition_list):
            new_node = new_partition_list[i][0]
            for old_node in old_community:
                for old_neighbor in self.neighbors[old_node]:
                    if old_neighbor not in old_community:
                        for j, other_old_community in enumerate(self.partition_list):
                            if i != j and old_neighbor in other_old_community:
                                new_neighbor = new_partition_list[j][0]
                                new_neighbors[new_node].add(new_neighbor)
                                new_neighbors[new_neighbor].add(new_node)

        self.partition_list = new_partition_list
        self.neighbors = new_neighbors

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


def average_weight_per_edge_cut(graph: nx.Graph, partition: Partition, val_lambda: float = 1) -> "float":
    node_mapping = {node: None for node in graph.nodes}
    community_data = {i: {'weight': 0.0, 'edges': 0, 'nodes': 0} for i in range(len(partition))}
    external_edge_weights = 0.0

    min_weight = None
    max_weight = None

    num_edges = 0
    num_cut_edges = 0
    for node1, node2, data in graph.edges(data=True):
        num_edges += 1

        if min_weight is None or data['weight'] < min_weight:
            min_weight = data['weight']
        if max_weight is None or data['weight'] > max_weight:
            max_weight = data['weight']

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

    # Internal term
    internal_degrees = 0
    for i, data in community_data.items():
        # squared sum of internal degrees
        internal_degrees += (2 * data['edges']) * (2 * data['edges'])
    internal_term = 0.01 * internal_degrees / (2 * num_edges)  # rescale degree

    # External term
    average_weight_per_cut_edge = external_edge_weights / num_cut_edges if num_cut_edges > 0 else 0.0
    external_term = log_sig(average_weight_per_cut_edge, max_weight)

    # good value for lambda ~0.01
    return external_term + val_lambda * internal_term


#########################
# Louvain Algortihm #####
#########################


def partition_louvain(graph: nx.Graph) -> "list[list[Any]]":
    # initial singleton partition
    partition = Partition.singleton_partition_from_graph(graph)

    while True:
        partition = _move_nodes(graph=graph, partition=partition)
        if len(partition.get_nodes()) == len(partition):
            break
        else:
            partition.aggregate()

    return partition.to_flat_partition()


def _move_nodes(
        graph: nx.Graph,
        partition: Partition,
        mod_func: Callable[[nx.Graph, Partition], float] = average_weight_per_edge_cut
) -> "Partition":
    mod_current = mod_func(graph, partition)
    nodes = partition.get_nodes()
    while True:
        mod_old = mod_current
        for node in nodes:
            max_partition = partition
            max_score = 0.0

            for new_partition in partition.get_partitions_for_node(node):
                new_score = mod_func(graph, new_partition) - mod_current
                if new_score > max_score:
                    # already ensures that the score is positive
                    max_partition = new_partition
                    max_score = new_score

            partition = max_partition
            # recalculate mod_current for the (possibly) changed partition
            mod_current += max_score

        if mod_old >= mod_current:
            # no change in modularity
            break

    return partition


########################
# Leiden Algorithm #####
########################


if __name__ == '__main__':
    test_graph = nx.Graph()
    test_graph.add_edge(0, 1, weight=random.random())
    test_graph.add_edge(1, 2, weight=random.random())
    test_graph.add_edge(2, 0, weight=random.random())
    test_graph.add_edge(0, 3, weight=random.random())
    test_graph.add_edge(3, 4, weight=random.random())

    part = Partition.singleton_partition_from_graph(test_graph)

    part2 = next(part.get_partitions_for_node(part.get_nodes()[3]))

    part2.aggregate()
    for k, v in part2.neighbors.items():
        print(f"{k}: {v}")


    # test_part = partition_louvain(test_graph)
    # print(test_part)
