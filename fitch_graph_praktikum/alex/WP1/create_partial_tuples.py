import pickle
import networkx as nx
import matplotlib.pyplot as plt
import fitch_graph_praktikum.util.lib as lib
import random
from fitch_graph_praktikum.nicolas.graph_io import load_relations


def create_partial_tuples(relation_dict, loss):
    # loss should be a number from 0,1 to 0,9 in 0,1 steps. I.e. 0,454 is not allowed, but 0,4 is.
    valid_loss_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    partial_tuples = {0: [], 1: [], 'd': []}

    if loss not in valid_loss_values:
        raise ValueError(
            "The value for loss must be one of those values: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 or 0.9 !")

    # symmetric edges (0 or 1):
    for key in (0, 1):
        for edge in relation_dict[key]:
            if edge[0] < edge[1]:
                if random.randint(0, 1) >= loss:
                    partial_tuples[key].append(edge)
                    partial_tuples[key].append((edge[1], edge[0]))
    # asymmetric edges ('d')
    for edge in relation_dict['d']:
        if random.randint(0, 1) >= loss:
            partial_tuples['d'].append(edge)

    return partial_tuples

if __name__ == '__main__':
    relations = load_relations(10, 5, 0)
    print(create_partial_tuples(relations,0.1))