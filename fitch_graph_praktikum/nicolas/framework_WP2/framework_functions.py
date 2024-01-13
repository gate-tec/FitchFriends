import networkx as nx
from fitch_graph_praktikum.alex.WP2.weight_scoring_for_partitioning import average_weight_scoring, sum_weight_scoring
from fitch_graph_praktikum.nicolas.framework_WP2.framework import (
    convert_to_bi_partition, average_weight_per_edge_cut, average_weight_per_edge_cut2, sum_weight_per_edge_cut
)
from fitch_graph_praktikum.nicolas.framework_WP2.random import partition_random
from fitch_graph_praktikum.alex.WP2.greedy_bipartitioning import greedy_bi_partition
from fitch_graph_praktikum.nicolas.framework_WP2.leiden import partition_leiden, LeidenLoopError
from fitch_graph_praktikum.nicolas.framework_WP2.louvain import partition_louvain


__all__ = [
    "bi_partition_random",
    "bi_partition_greedy_avg", "bi_partition_greedy_sum",
    "bi_partition_louvain_average_edge_cut", "bi_partition_louvain_sum_edge_cut",
    "bi_partition_louvain_average_edge_cut_mod",
    "bi_partition_leiden_average_edge_cut_gamma1_theta1", "bi_partition_leiden_sum_edge_cut_gamma1_theta1",
    "bi_partition_leiden_average_edge_cut_mod_gamma1_theta1",
    "bi_partition_leiden_average_edge_cut_gamma1_7_theta001", "bi_partition_leiden_sum_edge_cut_gamma1_7_theta001",
    "bi_partition_leiden_average_edge_cut_mod_gamma1_7_theta001"
]


#######################
# Random function #####
#######################


def bi_partition_random(graph: nx.Graph) -> "list[list]":
    return partition_random(graph=graph)


########################
# Greedy functions #####
########################


def bi_partition_greedy_avg(graph: nx.Graph) -> "list[list]":
    return greedy_bi_partition(graph=graph, score_func=average_weight_scoring)


def bi_partition_greedy_sum(graph: nx.Graph) -> "list[list]":
    return greedy_bi_partition(graph=graph, score_func=sum_weight_scoring)


#########################
# Louvain functions #####
#########################


def bi_partition_louvain_average_edge_cut(graph: nx.Graph) -> "list[list]":
    k_partition = partition_louvain(graph=graph, mod_func=average_weight_per_edge_cut2)

    return convert_to_bi_partition(graph=graph, k_partition=k_partition, mod_func=average_weight_scoring)


def bi_partition_louvain_average_edge_cut_mod(graph: nx.Graph) -> "list[list]":
    k_partition = partition_louvain(graph=graph, mod_func=average_weight_per_edge_cut)

    return convert_to_bi_partition(graph=graph, k_partition=k_partition, mod_func=average_weight_scoring)


def bi_partition_louvain_sum_edge_cut(graph: nx.Graph) -> "list[list]":
    k_partition = partition_louvain(graph=graph, mod_func=sum_weight_per_edge_cut)

    return convert_to_bi_partition(graph=graph, k_partition=k_partition, mod_func=sum_weight_scoring)


########################
# Leiden functions #####
########################

def bi_partition_leiden_average_edge_cut_gamma1_theta1(graph: nx.Graph) -> "list[list]":
    try:
        k_partition = partition_leiden(
            graph=graph,
            val_gamma=1.0, theta=1.0,
            mod_func=average_weight_per_edge_cut2
        )
    except LeidenLoopError as err:
        k_partition = err.last_common_partition.to_flat_partition()

    return convert_to_bi_partition(graph=graph, k_partition=k_partition, mod_func=average_weight_scoring)


def bi_partition_leiden_average_edge_cut_gamma1_7_theta001(graph: nx.Graph) -> "list[list]":
    try:
        k_partition = partition_leiden(
            graph=graph,
            val_gamma=1.0 / 7.0, theta=0.01,
            mod_func=average_weight_per_edge_cut2
        )
    except LeidenLoopError as err:
        k_partition = err.last_common_partition.to_flat_partition()

    return convert_to_bi_partition(graph=graph, k_partition=k_partition, mod_func=average_weight_scoring)


def bi_partition_leiden_average_edge_cut_mod_gamma1_theta1(graph: nx.Graph) -> "list[list]":
    try:
        k_partition = partition_leiden(
            graph=graph,
            val_gamma=1.0, theta=1.0,
            mod_func=average_weight_per_edge_cut
        )
    except LeidenLoopError as err:
        k_partition = err.last_common_partition.to_flat_partition()

    return convert_to_bi_partition(graph=graph, k_partition=k_partition, mod_func=average_weight_scoring)


def bi_partition_leiden_average_edge_cut_mod_gamma1_7_theta001(graph: nx.Graph) -> "list[list]":
    try:
        k_partition = partition_leiden(
            graph=graph,
            val_gamma=1.0 / 7.0, theta=0.01,
            mod_func=average_weight_per_edge_cut
        )
    except LeidenLoopError as err:
        k_partition = err.last_common_partition.to_flat_partition()

    return convert_to_bi_partition(graph=graph, k_partition=k_partition, mod_func=average_weight_scoring)


def bi_partition_leiden_sum_edge_cut_gamma1_theta1(graph: nx.Graph) -> "list[list]":
    try:
        k_partition = partition_leiden(
            graph=graph,
            val_gamma=1.0, theta=1.0,
            mod_func=sum_weight_per_edge_cut
        )
    except LeidenLoopError as err:
        k_partition = err.last_common_partition.to_flat_partition()

    return convert_to_bi_partition(graph=graph, k_partition=k_partition, mod_func=sum_weight_scoring)


def bi_partition_leiden_sum_edge_cut_gamma1_7_theta001(graph: nx.Graph) -> "list[list]":
    try:
        k_partition = partition_leiden(
            graph=graph,
            val_gamma=1.0 / 7.0, theta=0.01,
            mod_func=sum_weight_per_edge_cut
        )
    except LeidenLoopError as err:
        k_partition = err.last_common_partition.to_flat_partition()

    return convert_to_bi_partition(graph=graph, k_partition=k_partition, mod_func=sum_weight_scoring)
