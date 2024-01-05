import math
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import fitch_graph_praktikum.util.lib as lib
import pandas as pd
import random
from fitch_graph_praktikum.nicolas.graph_io import load_relations


def create_partial_tuples(relations_dict, loss=None, loose_absoluteNumber_of_elements=False):
    # loss should be a number from 0,1 to 0,9 in 0,1 steps. I.e. 0,454 is not allowed, but 0,4 is.
    global amount_to_keep
    valid_loss_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    partial_tuples = {0: [], 1: [], 'd': []}
    filter0 = list(filter(lambda edge: edge[0] < edge[1], relations_dict[0]))
    filter1 = list(filter(lambda edge: edge[0] < edge[1], relations_dict[1]))

    # since for keys 0 and 1 tuples are presented symmetrically, we only count half of their entries
    values_before = len(relations_dict[0]) / 2 + len(relations_dict[1]) / 2 + len(relations_dict['d'])

    is_absloss_used = False
    if loss is None and loose_absoluteNumber_of_elements is False:
        raise ValueError(
            "Either the relative loss or an absolute loss must be entered to make this function work"
        )
    if loss is not None:
        if loose_absoluteNumber_of_elements is False:
            if loss not in valid_loss_values:
                raise ValueError(
                    "The value for the relative loss must be one of those values: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 or 0.9 !")
            amount_to_keep = int(values_before * (1 - loss))
        else:
            if loss is not int or int(loss) != loss:
                raise ValueError(
                    "The absolute loss provided must be an integer e.g. 9 or et least equal to an integer e.g. 9.0!")
            amount_to_keep = int(values_before - loose_absoluteNumber_of_elements)
            is_absloss_used = True

    to_keep = random.sample(filter0 + filter1 + relations_dict['d'], amount_to_keep)
    for edge in to_keep:
        if edge in relations_dict[1]:
            partial_tuples[1].append(edge)
            partial_tuples[1].append((edge[1], edge[0]))
        elif edge in relations_dict[0]:
            partial_tuples[0].append(edge)
            partial_tuples[0].append((edge[1], edge[0]))
        else:
            partial_tuples['d'].append(edge)

    values_after = len(partial_tuples[0]) / 2 + len(partial_tuples[1]) / 2 + len(partial_tuples['d'])
    effRelLoss = round(((1 - (values_after / values_before)) * 100), 1)
    effAbsLoss = values_before - values_after
    # print(amount_to_keep)
    # print(values_after)
    if is_absloss_used is False:
        print("by creating partial tuples ", effRelLoss, "% information have been lost")
        return partial_tuples, effRelLoss
    else:
        print("by creating partial tuples ", effAbsLoss, " information on edges have been lost")
        return partial_tuples, effAbsLoss


def create_partial_tuples_cumulativeLoss(relations_dict, initial_relLoss, number_of_iterations=None):
    initial_values = len(relations_dict[0]) / 2 + len(relations_dict[1]) / 2 + len(relations_dict['d'])
    cumulative_loss_tuples = pd.DataFrame
    if number_of_iterations is None:
        number_of_iterations = math.floor(initial_values / (initial_values * initial_relLoss))
        if number_of_iterations == int(number_of_iterations):
            number_of_iterations = number_of_iterations - 1
        print("Since no number of iteration is provided, we will calculate the maximal amount of possible iterations (",
              number_of_iterations,")considering the cumulative loss")
        for i in range(number_of_iterations):

            pass

    if __name__ == '__main__':
        relations = load_relations(10, 5, 0)
        print(create_partial_tuples(relations, 0.7))
