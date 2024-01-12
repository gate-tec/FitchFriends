import random

import numpy

from fitch_graph_praktikum.alex.WP2.full_weighted_relations import generate_full_weighted_relations
from fitch_graph_praktikum.util.lib import partition_heuristic_scaffold
from typing import Literal
import networkx as nx


def average_weight_scoring(left: [], right: [], relations: dict[tuple[int, int], float]):
    score = 0
    counter = 0
    for l in left:
        for r in right:
            if (l, r) in relations:
                score += relations[(l, r)]
                counter += 1
    score = score / counter
    return score


def sum_weight_scoring(left: [], right: [], relations: dict[tuple[int, int], float]):
    score = 0
    for l in left:
        for r in right:
            if (l, r) in relations:
                score += relations[(l, r)]
    return score


def greedy_bipartition(graph: nx.Graph):
    nodes = [list(graph.nodes), []]
    uncut_edges = {(n1, n2): v['weight'] for n1, n2, v in graph.edges(data=True)}
    edges_cut = dict()

    score_new = 0

    cut = max(uncut_edges.keys(), key=uncut_edges.get)
    edges_cut.update(uncut_edges.pop(cut))
    print(nodes)
    nodes[1].append(nodes[0].pop(cut[0]))
    print(nodes)
    score_new = uncut_edges[cut]
    print("_initial cut_edge: ", cut, " score_new = ", score_new)

    neighbours = set(graph.neighbors(cut[0]))
    neighbours.remove(cut[1])

    for n in neighbours:
        neighbor_cut = [cut[0]][n]
        score_new = score_new + graph[neighbor_cut]['weight']
        edges_cut.update(uncut_edges.pop(neighbor_cut))

    score_old = score_new - 1

    while score_new > score_old:
        nodes_backup = nodes
        score_old = score_new

        cut = None
        neighbours = set()
        for n in nodes[1]:
            neighbours.update({neighbour for neighbour in list(graph.neighbors(n))})

        for ue in uncut_edges:
            if ue[1] not in nodes[1] and ue[0] not in nodes[1]:
                if ue[0] in neighbours or ue[1] in neighbours:
                    if cut is None or uncut_edges[cut] > uncut_edges[ue]:
                        cut = ue
                        print("potential cut: ", cut)

        neighbours = set(graph.neighbors(cut[0]))
        for n in nodes[1]:
            neighbours.remove(n)
        for n in neighbours:
            neighbor_cut = [cut[0]][n]
            score_ = score_new + graph[neighbor_cut]['weight']
            edges_cut.update(uncut_edges.pop(neighbor_cut))


            max(uncut_edges.keys(), key=uncut_edges.get)

    return


# test = partition_heuristic_scaffold({}, {}, {}, [], partition_function=0, scoring_function=0)


if __name__ == '__main__':
    # print(numpy.random.normal)
    # test_relations = {0: [(1, 2), (2, 1)], 1: [(0, 1), (1, 0)], 'd': [(0, 2)]}
    # test_weights_sym = generate_full_weighted_relations(3, test_relations, numpy.random.normal, [3, 0.75],
    #                                                     numpy.random.normal, [1, 0.5], symmetric=True)
    # test_weights_asym = generate_weighted_relations(test_relations, numpy.random.normal, [3, 0.75], symmetric=False)
    # print(test_weights_sym)
    # for k, v in test_weights_sym.items():
    #     print(f"{k}: {v}")
    #
    # test = partition_heuristic_scaffold({}, {}, {}, [], partition_function=0, scoring_function=0)

    edges = [(0, 1, 2.0), (2, 0, 3.0), (1, 2, 0.5), (0, 3, 0.9), (1, 3, 1.5), (2, 3, 2.0)]
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    print(G[0][1])
