import random

import networkx as nx


#################################
# Random partition function #####
#################################


def partition_random(graph: nx.Graph) -> "list[list]":
    graph_nodes = [x for x in graph.nodes]
    partition = [[graph_nodes[0]], [graph_nodes[1]]]
    for i in range(2, len(graph_nodes)):
        if random.random() >= 0.5:
            partition[0].append(graph_nodes[i])
        else:
            partition[1].append(graph_nodes[i])

    return partition
