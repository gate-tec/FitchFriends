import random

import networkx as nx


#################################
# Random partition function #####
#################################


def partition_random(graph: nx.Graph) -> "list[list]":
    partition = [[], []]
    for node in graph.nodes:
        if random.random() > 0.5:
            partition[0].append(node)
        else:
            partition[1].append(node)

    return partition
