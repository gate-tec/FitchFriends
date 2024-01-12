import networkx as nx
import copy
import numpy as np

from fitch_graph_praktikum.nicolas.framework_WP2.framework import (
    BasePartition, average_weight_per_edge_cut2
)

from typing import TYPE_CHECKING, Optional, Callable, Literal

if TYPE_CHECKING:
    from typing import Any, Dict, Generator


__all__ = ["LouvainPartition", "partition_louvain"]


#################################
# Louvain Partition Variant #####
#################################


class LouvainPartition(BasePartition):

    def __init__(self,
                 partition_list: list[list[frozenset]],
                 nodes: list[frozenset]):
        super().__init__(partition_list, nodes)

    @classmethod
    def singleton_partition_from_graph(cls, graph: nx.Graph) -> "LouvainPartition":
        """Initialize singleton partition from a list of nodes."""
        # convert all node representations to frozensets
        relabeled_graph = nx.relabel_nodes(graph, {node: frozenset({node}) for node in graph.nodes})
        # construct singleton partition
        return cls([[node] for node in relabeled_graph.nodes], [node for node in relabeled_graph.nodes])

    def get_partitions_for_node(self, node) -> "Generator[LouvainPartition]":
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
            yield LouvainPartition(new_partition_list, self.nodes)
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
            yield LouvainPartition(new_partition_list, self.nodes)

#########################
# Louvain Algortihm #####
#########################


def partition_louvain(
        graph: nx.Graph,
        mod_func: Callable[[nx.Graph, BasePartition], float] = average_weight_per_edge_cut2
) -> "list[list[Any]]":
    # initial singleton partition
    partition = LouvainPartition.singleton_partition_from_graph(graph)

    while True:
        partition = _move_nodes(graph=graph, partition=partition, mod_func=mod_func)
        if len(partition.get_nodes()) == len(partition):
            break
        else:
            partition.aggregate()

    return partition.to_flat_partition()


def _move_nodes(
        graph: nx.Graph,
        partition: LouvainPartition,
        mod_func: Callable[[nx.Graph, BasePartition], float] = average_weight_per_edge_cut2
) -> "LouvainPartition":
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
