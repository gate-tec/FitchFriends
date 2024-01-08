import random

import networkx as nx
import copy

from typing import TYPE_CHECKING, Optional, Callable, Literal

if TYPE_CHECKING:
    from typing import Any, Dict, Generator


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
        neighbors = {node: dict(relabeled_graph[node]) for node in relabeled_graph.nodes}
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
        self.partition_list = [[frozenset({node for entry in community for node in entry})] for community in self.partition_list]
        self.nodes = [community[0] for community in self.partition_list]
        # recalculate neighbors (using average weight per aggregated edges)
        # TODO

    def get_nodes(self):
        return self.nodes

    def to_flat_partition(self):
        return [[node for entry in community for node in entry] for community in self.partition_list]

    def __len__(self):
        return len(self.partition_list)


def average_weight_per_edge_cut(graph: nx.Graph, partition: Partition) -> "float":
    return 1.0


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
