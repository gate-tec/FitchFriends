import networkx as nx
import copy
import numpy as np

from fitch_graph_praktikum.nicolas.framework_WP2.framework import (
    BasePartition, average_weight_per_edge_cut2
)

from typing import TYPE_CHECKING, Optional, Callable, Literal

if TYPE_CHECKING:
    from typing import Any, Dict, Generator


__all__ = ["LeidenPartition", "partition_leiden", "LeidenLoopError"]


################################
# Leiden Partition Variant #####
################################


class LeidenPartition(BasePartition):

    def __init__(self,
                 partition_list: list[list[frozenset]],
                 nodes: list[frozenset],
                 neighbors: dict[frozenset, dict[frozenset, dict[Literal['weight'], float]]]):
        super().__init__(partition_list, nodes)
        self.neighbors = neighbors

    @classmethod
    def singleton_partition_from_graph(cls, graph: nx.Graph) -> "LeidenPartition":
        """Initialize singleton partition from a list of nodes."""
        # convert all node representations to frozensets
        relabeled_graph = nx.relabel_nodes(graph, {node: frozenset({node}) for node in graph.nodes})
        # get relations and weights from base graph
        neighbors = {node: set(relabeled_graph[node].keys()) for node in relabeled_graph.nodes}
        # construct singleton partition
        return cls([[node] for node in relabeled_graph.nodes], [node for node in relabeled_graph.nodes], neighbors)

    def to_singleton_partition(self) -> "LeidenPartition":
        partition_list = [[node] for node in self.get_nodes()]
        return LeidenPartition(partition_list, self.nodes, self.neighbors)

    def aggregate(self):
        new_partition_list = [
            [frozenset({node for entry in community for node in entry})] for community in self.partition_list
        ]
        self.nodes = [community[0] for community in new_partition_list]
        # recalculate neighbors
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

    def aggregate_from_refined_partition(self, refined_partition):
        """

        Parameters
        ----------
        refined_partition: LeidenPartition
        """
        refined_partition.aggregate()
        aggregated_partition_list = refined_partition.partition_list

        new_partition_list = [[] for _ in range(len(self.partition_list))]
        for aggregated_community in aggregated_partition_list:
            node = aggregated_community[0]  # only one node due to aggregation
            first_base_node = list(node)[0]  # extract first base node from aggregated node
            for i, community in enumerate(self.partition_list):
                is_in_community = False
                for entry in community:
                    if first_base_node in entry:
                        is_in_community = True
                        break
                if is_in_community:
                    new_partition_list[i].append(node)
                    break

        self.nodes = refined_partition.nodes
        self.partition_list = new_partition_list
        self.neighbors = refined_partition.neighbors

    def get_partitions_for_node(self,
                                node,
                                subset: list[frozenset] = None,
                                len_subset: int = None,
                                subset_mapping: dict[tuple[frozenset, frozenset], int] = None,
                                val_gamma: float = 1.0) -> "Generator[LeidenPartition]":
        # prepare list of communities to check
        partition_range = None
        if subset is not None:
            partition_range = []
            if len_subset is None or subset_mapping is None:
                raise ValueError

        # find community of node
        community_index = -1
        for i in range(len(self.partition_list)):
            if node in self.partition_list[i]:
                community_index = i
                if subset is None:
                    break
            elif partition_range is not None:
                # find well-connected communities in subset
                community = self.partition_list[i]

                if community[0] not in subset:
                    # quick preliminary check
                    continue

                # calculate: subset \ community
                len_community = 0
                community_complement = []
                for other_node in subset:
                    if other_node in community:
                        len_community += 1
                    else:
                        community_complement.append(other_node)

                if len_subset != len_community + len(community_complement):
                    # community can't be contained in subset
                    continue
                num_edges = sum(subset_mapping[(node1, node2)] for node1 in community for node2 in community_complement)

                if num_edges >= val_gamma * len_community * len(community_complement):
                    # community is well-connected within subset
                    partition_range.append(i if community_index == -1 else i - 1)  # fix index as community for node will be deleted later

        # sanity check
        if community_index == -1:
            raise ValueError("Corresponding community not found")

        partial_partition_list = copy.deepcopy(self.partition_list)
        partial_partition_list[community_index].remove(node)

        if len(partial_partition_list[community_index]) > 0:
            if subset is not None:
                # sanity check
                raise ValueError
            # moving node in its own community yields a different partition
            new_partition_list = copy.deepcopy(partial_partition_list)
            new_partition_list.append([node])
            yield LeidenPartition(new_partition_list, self.nodes, self.neighbors)
        else:
            # remove empty community
            del partial_partition_list[community_index]
            community_index = -1

        if partition_range is None:
            partition_range = range(len(partial_partition_list))

        for i in partition_range:
            if i == community_index:
                # skip old community
                continue
            new_partition_list = copy.deepcopy(partial_partition_list)
            new_partition_list[i].append(node)
            yield LeidenPartition(new_partition_list, self.nodes, self.neighbors)

    def get_hash(self):
        # [[node for entry in community for node in entry] for community in self.partition_list]
        return hash(
            tuple(sorted(tuple(sorted(tuple(sorted(node for node in entry)) for entry in community)) for community in self.partition_list))
        )


########################
# Leiden Algorithm #####
########################


class LeidenLoopError(Exception):
    """Raised when execution of Leiden alg was forced to stop after detecting a loop."""

    def __init__(self, last_common_partition: LeidenPartition):
        super().__init__('Leiden algorithm was trapped in a loop.')
        self.last_common_partition = last_common_partition


def partition_leiden(
        graph: nx.Graph,
        val_gamma: float = 1.0,
        theta: float = 1.0,
        mod_func: Callable[[nx.Graph, BasePartition], float] = average_weight_per_edge_cut2
) -> "list[list[Any]]":
    # initial singleton partition
    partition = LeidenPartition.singleton_partition_from_graph(graph)
    hash_map = {partition.get_hash(): 1}

    while True:
        partition = _move_nodes_fast(graph=graph, partition=partition, mod_func=mod_func)
        if len(partition.get_nodes()) == len(partition):
            break
        else:
            partition_refined = _refine_partition(
                graph=graph, partition=partition, val_gamma=val_gamma, theta=theta, mod_func=mod_func
            )
            partition.aggregate_from_refined_partition(partition_refined)

            hash_code = partition.get_hash()
            if hash_code not in hash_map.keys():
                hash_map[hash_code] = 1
            elif hash_map[hash_code] < 3:
                hash_map[hash_code] += 1
            else:
                raise LeidenLoopError(partition)

    return partition.to_flat_partition()


def _move_nodes_fast(
        graph: nx.Graph,
        partition: LeidenPartition,
        mod_func: Callable[[nx.Graph, BasePartition], float] = average_weight_per_edge_cut2
) -> "LeidenPartition":
    queue = [node for node in partition.get_nodes()]
    in_queue = {node: True for node in partition.get_nodes()}
    mod_current = mod_func(graph, partition)

    while queue:
        node = queue.pop(0)
        in_queue[node] = False

        max_partition = partition
        max_score = 0.0

        for new_partition in partition.get_partitions_for_node(node):
            new_score = mod_func(graph, new_partition) - mod_current
            if new_score > max_score:
                # already ensures that the score is positive
                max_partition = new_partition
                max_score = new_score

        if max_score > 0:
            partition = max_partition
            # recalculate mod_current for the changed partition
            mod_current += max_score

            # re-locate community of node
            new_community = None
            for community in partition.partition_list:
                if node in community:
                    new_community = community
                    break

            for neighbor_node in partition.neighbors[node]:
                if not in_queue[neighbor_node] and neighbor_node not in new_community:
                    queue.append(neighbor_node)
                    in_queue[neighbor_node] = True

    return partition


def _refine_partition(
        graph: nx.Graph,
        partition: LeidenPartition,
        val_gamma: float = 1.0,
        theta: float = 1.0,
        mod_func: Callable[[nx.Graph, BasePartition], float] = average_weight_per_edge_cut2
) -> "LeidenPartition":
    partition_refined = partition.to_singleton_partition()
    for community in partition.partition_list:
        partition_refined = _merge_nodes_subset(
            graph=graph,
            partition=partition_refined,
            subset=community,
            val_gamma=val_gamma,
            theta=theta,
            mod_func=mod_func
        )

    return partition_refined


def _merge_nodes_subset(
        graph: nx.Graph,
        partition: LeidenPartition,
        subset: list[frozenset],
        val_gamma: float = 1.0,
        theta: float = 1.0,
        mod_func: Callable[[nx.Graph, BasePartition], float] = average_weight_per_edge_cut2
) -> "LeidenPartition":
    # calculate lens
    lens = {}
    len_subset = 0
    for node in subset:
        lens[node] = len(node)
        len_subset += lens[node]

    # Dynamic programming for finding well-connected nodes in subset
    mapping = {(node1, node2): None for node1 in subset for node2 in subset if node1 != node2}
    # Line 34 of pseudocode:
    set_r = []
    for i, node1 in enumerate(subset):
        num_edges = 0
        right_term = val_gamma * lens[node1] * (len_subset - lens[node1])
        for j, node2 in enumerate(subset):
            if i == j:
                continue
            if mapping[(node1, node2)] is None:
                edges = sum(1 for vertex1 in node1 for vertex2 in node2 if graph.has_edge(vertex1, vertex2))
                mapping[(node1, node2)] = edges
                mapping[(node2, node1)] = edges
            else:
                edges = mapping[(node1, node2)]
            num_edges += edges

        if num_edges >= right_term:
            set_r.append(node1)

    mod_current = mod_func(graph, partition)

    # Line 35 of pseudocode:
    for node in set_r:
        old_community = None
        for community in partition.partition_list:
            if node in community:
                old_community = community
                break
        if old_community is None:
            # sanity check
            raise KeyError
        # Line 36 of pseudocode:
        if len(old_community) == 1:
            # node is in singleton community

            max_partition = partition
            max_score = 0.0
            max_prob = 0.0

            for new_partition in partition.get_partitions_for_node(
                node,
                subset=subset,
                len_subset=len_subset,
                subset_mapping=mapping,
                val_gamma=val_gamma
            ):
                new_score = mod_func(graph, new_partition) - mod_current
                if new_score <= 0.0:
                    continue
                prob = np.random.exponential(1 / theta * new_score)
                if prob > max_prob:
                    # already ensures that the score is positive
                    max_partition = new_partition
                    max_score = new_score
                    max_prob = prob

            partition = max_partition
            # recalculate mod_current for the changed partition
            mod_current += max_score

    return partition