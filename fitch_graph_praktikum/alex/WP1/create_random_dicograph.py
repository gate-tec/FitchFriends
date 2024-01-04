import matplotlib.pyplot as plt
import networkx as nx
import random


def create_random_dicograph(n):
    # l represents the number of disjoint_nodes within the tree
    operations = [0, 1, 'u']
    disjoint_nodes = []
    dicograph_notation = []
    dicograph = nx.DiGraph()
    for i in range(0, n):
        dicograph.add_node(i)
        disjoint_nodes.append([i])
    # print(disjoint_nodes)

    while len(disjoint_nodes) != 1:
        a, b = random.sample(disjoint_nodes, 2)
        operator = random.choice(operations)

        if operator == 1:
            for node_in_a in a:
                for nodes_in_b in b:
                    dicograph.add_edge(node_in_a, nodes_in_b)
                    dicograph.add_edge(nodes_in_b, node_in_a)

        if operator == 'u':
            for node_in_a in a:
                for nodes_in_b in b:
                    dicograph.add_edge(node_in_a, nodes_in_b)
        a += b

        disjoint_nodes.remove(b)
    return dicograph


if __name__ == '__main__':
    graph = create_random_dicograph(8)
    nx.draw(graph)
    plt.show()
