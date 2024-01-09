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


def average_weight_per_edge_cut(graph: nx.Graph, partition: BasePartition, val_lambda: float = 0.6) -> "float":
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
    partition = LouvainPartition.singleton_partition_from_graph(graph)

    while True:
        partition = _move_nodes(graph=graph, partition=partition)
        if len(partition.get_nodes()) == len(partition):
            break
        else:
            partition.aggregate()

    return partition.to_flat_partition()


def _move_nodes(
        graph: nx.Graph,
        partition: LouvainPartition,
        mod_func: Callable[[nx.Graph, BasePartition], float] = average_weight_per_edge_cut
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


########################
# Leiden Algorithm #####
########################


def partition_leiden(graph: nx.Graph, val_gamma: float = 1.0, theta: float = 1.0) -> "list[list[Any]]":
    # initial singleton partition
    partition = LeidenPartition.singleton_partition_from_graph(graph)

    while True:
        partition = _move_nodes_fast(graph=graph, partition=partition)
        if len(partition.get_nodes()) == len(partition):
            break
        else:
            partition_refined = _refine_partition(graph=graph, partition=partition, val_gamma=val_gamma, theta=theta)
            partition.aggregate_from_refined_partition(partition_refined)

    return partition.to_flat_partition()


def _move_nodes_fast(
        graph: nx.Graph,
        partition: LeidenPartition,
        mod_func: Callable[[nx.Graph, BasePartition], float] = average_weight_per_edge_cut
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
        theta: float = 1.0) -> "LeidenPartition":
    partition_refined = partition.to_singleton_partition()
    for community in partition.partition_list:
        partition_refined = _merge_nodes_subset(
            graph=graph,
            partition=partition_refined,
            subset=community,
            val_gamma=val_gamma,
            theta=theta
        )

    return partition_refined


def _merge_nodes_subset(
        graph: nx.Graph,
        partition: LeidenPartition,
        subset: list[frozenset],
        val_gamma: float = 1.0,
        theta: float = 1.0,
        mod_func: Callable[[nx.Graph, BasePartition], float] = average_weight_per_edge_cut) -> "LeidenPartition":
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

            prev_score = float(np.exp(0))
            partition_mapping = [(prev_score, partition, 0.0)]

            # Line 37 of pseudocode
            for new_partition in partition.get_partitions_for_node(
                node,
                subset=subset,
                len_subset=len_subset,
                subset_mapping=mapping,
                val_gamma=val_gamma
            ):
                new_score = mod_func(graph, new_partition) - mod_current
                if new_score >= 0.0:
                    prev_score = prev_score + float(np.exp(1/theta * new_score))
                    partition_mapping.append((prev_score, new_partition, new_score))

            # Line 38 of pseudocode
            selected_score = random.random() * prev_score
            for score, new_partition, new_score in partition_mapping:
                if selected_score <= score:
                    # Line 39 of pseudocode
                    partition = new_partition
                    mod_current += new_score

    return partition


if __name__ == '__main__':
    test_graph = nx.Graph()
    test_graph.add_edge(0, 1, weight=random.random())
    test_graph.add_edge(1, 2, weight=random.random())
    test_graph.add_edge(2, 0, weight=random.random())
    test_graph.add_edge(0, 3, weight=random.random())
    test_graph.add_edge(3, 4, weight=random.random())

    # test_part = partition_louvain(test_graph)
    # print(test_part)

    # [[1, 2], [0, 3, 4]]
    # [(0, 1, {'weight': 0.8280202664719893}), (0, 2, {'weight': 0.8468783050487141}), (0, 3, {'weight': 0.005807499470777189}), (1, 2, {'weight': 0.79559227057101}), (3, 4, {'weight': 0.09581249221137933})]

    # [[0, 1, 2, 3], [4]]
    # [(0, 1, {'weight': 0.20328625639809128}), (0, 2, {'weight': 0.16334338262731352}), (0, 3, {'weight': 0.4519652680782352}), (1, 2, {'weight': 0.4951572675695667}), (3, 4, {'weight': 0.7764183662304466})]

    test_part = partition_leiden(test_graph, val_gamma=1.0/7.0)
    print(test_part)
    print(test_graph.edges(data=True))
