import numpy.random

from fitch_graph_praktikum.util.lib import generate_weights
from fitch_graph_praktikum.nicolas.graph_io import load_relations
import numpy as np


def generate_full_weighted_relations(number_of_nodes: int, relations: dict, distribution_TP, parameters_TP, distribution_FP,
                                parameters_FP, symmetric=True):
    """generates weighted relations in the format of a WeightedRelationDictionary provided in nicolas/typing """
    nodes = [x for x in range(number_of_nodes)]
    weighted_relations = {0: {}, 1: {}, 'd': {}}
    relations_complement = {0: [], 1: [], 'd': []}
    operations = [0, 1, 'd']
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if i == j:
                continue
            if (i, j) not in relations[0]:
                relations_complement[0].append((i, j))
            if (i, j) not in relations[1]:
                relations_complement[1].append((i, j))
            if (i, j) not in relations['d']:
                relations_complement['d'].append((i, j))

    for i in range(3):
        weights_original = generate_weights(relations[operations[i]], distribution_TP, parameters_TP, symmetric)
        weights_complement = generate_weights(relations_complement[operations[i]], distribution_FP, parameters_FP,
                                              symmetric)

        weighted_relations[operations[i]] = {k: round(v, 3) for k, v in
                                             {**weights_original, **weights_complement}.items()}
    # complement
    return weighted_relations


if __name__ == '__main__':
    # print(numpy.random.normal)
    test_relations = {0: [(1, 2), (2, 1)], 1: [(0, 1), (1, 0)], 'd': [(0, 2)]}
    test_weights_sym = generate_full_weighted_relations(3, test_relations, numpy.random.normal, [3, 0.75],
                                                   numpy.random.normal, [1, 0.5], symmetric=True)
    # test_weights_asym = generate_weighted_relations(test_relations, numpy.random.normal, [3, 0.75], symmetric=False)
    # print(test_weights_sym)
    for k, v in test_weights_sym.items():
        print(f"{k}: {v}")
