import random
import copy
from typing import Callable

import networkx as nx


def _return_neighbours_new_part(graph: nx.Graph, node_list) -> "set":
    neighbours = set()
    for n in node_list[1]:
        neighbours.update({neighbour for neighbour in list(graph.neighbors(n)) if neighbour not in node_list[1]})
    return neighbours


def _best_greedy_move(
        graph: nx.Graph,
        nodelist: list,
        edges: dict[tuple, float],
        score_func: Callable[[list, list, dict[tuple[int, int], float]], float],
        reference_score: float = None):
    """to bipartition the graphs nodes, we check first, whether the selected node lies in the neighbourhood of
    nodelist[1]. This is logically correct to avoid partitions into more than two communities. Also, it would never
    appear for nodelist[0] to contain no neighbors of nodelist[1] - since this would mean, all nodes would have been
    moved to nodelist[1] - which would worsen the score to zero again"""
    neighbours = _return_neighbours_new_part(graph=graph, node_list=nodelist)
    nodes_updated = False

    for n0 in neighbours:
        greedy_test = copy.deepcopy(nodelist)
        greedy_test[1].append(n0)
        greedy_test[0].remove(n0)
        greedy_test_score = score_func(greedy_test[0], greedy_test[1], edges)

        if reference_score is None:
            # update reference score.
            reference_score = greedy_test_score - 1
        if greedy_test_score > reference_score:
            # update reference score.
            reference_score = greedy_test_score
            nodes_updated = greedy_test

    if nodes_updated is not False:
        return nodes_updated
    else:
        return False


def _adjust_greedy(
        graph: nx.Graph,
        nodelist: list,
        edges: dict[tuple, float],
        score_func: Callable[[list, list, dict[tuple[int, int], float]], float],
        reference_score: float):
    """re-partitions one node from nodelist[1] back to nodelist[0] except for the last node
    and based on  this reperforms the best greedy move"""

    greedy_adjusted = False

    index_last_node_to_move = len(nodelist[1]) - 1
    # probiere alle elemente bis auf das letzte aus:
    for i in range(index_last_node_to_move):
        nodes_to_rollback = copy.deepcopy(nodelist)
        node_to_move = nodes_to_rollback[1][i]
        nodes_to_rollback[0].append(node_to_move)
        nodes_to_rollback[1].remove(node_to_move)

        nodes_adjusted = _best_greedy_move(
            graph=graph, nodelist=nodes_to_rollback, edges=edges, score_func=score_func, reference_score=reference_score
        )
        if nodes_adjusted is not False:
            score_adjusted = score_func(nodes_adjusted[0], nodes_adjusted[1], edges)
            reference_score = score_adjusted
            greedy_adjusted = nodes_adjusted

    return greedy_adjusted


###################################################################################################################


def greedy_bi_partition(graph: nx.Graph, score_func: Callable[[list, list, dict[tuple[int, int], float]], float]):

    nodes = [list(graph.nodes), []]
    edges = {(n1, n2): v['weight'] for n1, n2, v in graph.edges(data=True)}
    edges.update({(n2, n1): v['weight'] for n1, n2, v in graph.edges(data=True)})

    starter = random.choice(nodes[0])
    nodes[0].remove(starter)
    nodes[1].append(starter)

    score = score_func(nodes[0], nodes[1], edges)

    score_old = score - 1
    while score > score_old:
        score_old = score

        nodes_updated = _best_greedy_move(
            graph=graph, nodelist=nodes, edges=edges, score_func=score_func, reference_score=score_old
        )
        if nodes_updated is False:
            nodes_updated = _adjust_greedy(
                graph=graph, nodelist=nodes, edges=edges, score_func=score_func, reference_score=score_old
            )

        if nodes_updated is False:
            continue

        else:
            nodes = nodes_updated
            score = score_func(nodes[0], nodes[1], edges)

    return nodes
