# Authors:
#   - Nicolas Handke    https://github.com/gate-tec

from fitch_graph_praktikum.nicolas.functions_partial_tuple import delete_information
from fitch_graph_praktikum.nicolas.tree_functions import create_tree, construct_random_cograph
from fitch_graph_praktikum.nicolas.graph_io import load_relations
from fitch_graph_praktikum.util.lib import cotree_to_rel, rel_to_fitch, check_fitch_graph, graph_to_rel

import networkx as nx
import matplotlib.pyplot as plt


##############################
# Simulated Fitch-Graphs #####
##############################

num_leaves = 10
relations = load_relations(num_leaves, 0, 0)
nodes = [str(x) for x in range(num_leaves)]

# Delete ~50% relations
reduced_relations, part = delete_information(relations, 50)
# print(reduced_relations)

# Check if Fitch-graph properties are lost
fitch_graph = rel_to_fitch(reduced_relations, nodes)
print(check_fitch_graph(fitch_graph))

#################################
# Random top-down dicotrees #####
#################################

nodes = [x for x in range(num_leaves)]
tree = create_tree(nodes)
# nx.draw(tree)
# plt.draw()
# plt.show()

relations = cotree_to_rel(tree)

# Check if it is by chance a Fitch-graph
fitch_graph = rel_to_fitch(relations, nodes)
print(check_fitch_graph(fitch_graph))

########################
# Random dicograph #####
########################

nodes = [x for x in range(num_leaves)]
graph = construct_random_cograph(nodes)
# nx.draw(graph)
# plt.show()

relations = graph_to_rel(graph)

# Check if it is by chance a Fitch-graph
fitch_graph = rel_to_fitch(relations, nodes)
# nx.draw(fitch_graph)
# plt.show()
print(check_fitch_graph(fitch_graph))
