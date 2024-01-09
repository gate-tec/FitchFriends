from create_partial_tuples import create_partial_tuples
from create_random_dicograph import create_random_dicograph
from fitch_graph_praktikum.util.helper_functions import NoSatRelation, NotFitchSatError
from fitch_graph_praktikum.util.lib import graph_to_rel, algorithm_one
import networkx as nx
import math


def random_dicograph_to_partialtuple(nodes, loss):
    """loss must be a number of the following [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    leaves as an integer represents the amount of nodes of the dicograph"""

    G = create_random_dicograph(nodes)
    relations = graph_to_rel(G)
    random_partial_tuples = create_partial_tuples(relations, loss)
    return random_partial_tuples

