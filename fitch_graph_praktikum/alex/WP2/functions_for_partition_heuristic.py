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
    nodes = list(graph.nodes)
    edges = {(n1, n2): v['weight'] for n1, n2, v in graph.edges(data=True)}


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

        edges = [(0, 1), (2, 0), (1, 2), (0, 3), (1, 3), (2, 3)]
        G = nx.Graph()
        G.add_edges_from(edges)
        print(G.edges)
