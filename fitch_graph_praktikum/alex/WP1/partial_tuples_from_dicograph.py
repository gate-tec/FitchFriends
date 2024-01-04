from create_partial_tuples import create_partial_tuples
from create_random_dicotree import create_random_dicograph
from fitch_graph_praktikum.util.lib import graph_to_rel

if __name__ == '__main__':
    amount_of_leaves = 10
    loss = 0,1
    G = create_random_dicograph(amount_of_leaves)
    relations = graph_to_rel(G)
    random_partial_tuples = create_partial_tuples(relations,)