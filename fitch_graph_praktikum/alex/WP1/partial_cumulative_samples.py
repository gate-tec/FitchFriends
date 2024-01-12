import math
from typing import Literal
from fitch_graph_praktikum.alex.WP1.create_partial_tuples import create_partial_tuples
from fitch_graph_praktikum.nicolas.graph_io import load_relations, save_dataframe
from fitch_graph_praktikum.util.helper_functions import NotFitchSatError, NoSatRelation
from fitch_graph_praktikum.util.lib import algorithm_one
import pandas as pd


def create_partial_tuples_cumulativeLoss(number_of_nodes: int,
                                         x_options: Literal[0, 1, 2, 3, 4, 5],
                                         x_instance,
                                         relations_dict: dict,
                                         initial_relLoss: float,
                                         number_of_iterations=None):
    """Wrapper-Function for generating random partial tuples on a cumulative (common) basis"""

    relations_last_iteration = relations_dict
    initial_values = len(relations_dict[0]) / 2 + len(relations_dict[1]) / 2 + len(relations_dict['d'])
    abs_Loss = math.ceil(initial_values * initial_relLoss)
    cumulative_loss_tuples = []
    total_loss = 0

    if number_of_iterations is None:
        number_of_iterations = initial_values / abs_Loss
        if number_of_iterations == int(number_of_iterations):
            number_of_iterations = number_of_iterations - 1
        else:
            number_of_iterations = math.floor(number_of_iterations)
        print("Since no number of iteration is provided, we will calculate the maximal amount of possible iterations (",
              number_of_iterations, ")considering the cumulative loss")

    # initial sample for further reference:
    try:
        is_fitchtree = algorithm_one(relations_dict, list(range(number_of_nodes)), order=(0, 1, 2))
        fitchsat = True
    except (NoSatRelation, NotFitchSatError):
        fitchsat = False

    sample = {'Nodes': number_of_nodes,
              'x_options': x_options,
              'Sample': x_instance,
              'Nominal_Loss': 0,
              'Effective_Loss': 0,
              'Is_Fitch_Sat': fitchsat, 'Relations': relations_dict}
    cumulative_loss_tuples.append(sample)

    for i in range(number_of_iterations):
        sample_relations, loss_in_iteration = create_partial_tuples(relations_last_iteration, abs_Loss,
                                                                    loose_absoluteNumber_of_elements=True)
        total_loss = total_loss + loss_in_iteration

        try:
            is_fitchtree = algorithm_one(sample_relations, list(range(number_of_nodes)), order=(0, 1, 2))
            fitchsat = True
        except (NoSatRelation, NotFitchSatError):
            fitchsat = False
        sample = {'Nodes': number_of_nodes, 'x_options': x_options, 'Sample': x_instance,
                  'Nominal_Loss': round((i + 1) * initial_relLoss * 100),
                  'Effective_Loss': int((total_loss / initial_values) * 100),
                  'Is_Fitch_Sat': fitchsat, 'Relations': sample_relations}

        cumulative_loss_tuples.append(sample)
        relations_last_iteration = sample_relations

    return cumulative_loss_tuples


if __name__ == '__main__':
    test_initial_relation = load_relations(15, 5, 9)

    print("no of information",
          len(test_initial_relation[0]) / 2 + len(test_initial_relation[1]) / 2 + len(test_initial_relation['d']))
    sample = create_partial_tuples_cumulativeLoss(15, 5, 9, test_initial_relation, 0.1)
    sample_df = pd.DataFrame(sample)
    print(sample_df)
    sample_df.to_csv("cumulative_test_sample.tsv", sep="\t")
