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
    filter0 = list(filter(lambda edge: edge[0] < edge[1], relation_dict[0]))
    filter1 = list(filter(lambda edge: edge[0] < edge[1], relation_dict[1]))

    # since for keys 0 and 1 tuples are presented symmetrically, we only count half of their entries
    values_before = len(relation_dict[0]) / 2 + len(relation_dict[1]) / 2 + len(relation_dict['d'])
    if loss not in valid_loss_values:
        raise ValueError(
            "The value for loss must be one of those values: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 or 0.9 !")
    amount_to_keep = int(values_before * (1 - loss))
    to_keep = random.sample(filter0 + filter1 + relation_dict['d'], amount_to_keep)
    for edge in to_keep:
        if edge in relation_dict[1]:
            partial_tuples[1].append(edge)
            partial_tuples[1].append((edge[1], edge[0]))
        elif edge in relation_dict[0]:
            partial_tuples[0].append(edge)
            partial_tuples[0].append((edge[1], edge[0]))
        else:
            partial_tuples['d'].append(edge)

    values_after = len(partial_tuples[0]) / 2 + len(partial_tuples[1]) / 2 + len(partial_tuples['d'])
    information_loss_percent = round((1 - (values_after / values_before))*100)
    print(amount_to_keep)
    print(values_after)
    print("by creating partial tuples ", information_loss_percent, "% information have been lost")

    return partial_tuples, information_loss_percent


if __name__ == '__main__':
    relations = load_relations(10, 5, 0)
    print(create_partial_tuples(relations, 0.7))
