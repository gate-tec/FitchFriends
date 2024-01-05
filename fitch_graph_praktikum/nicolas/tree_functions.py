import random
from typing import List, Any

import networkx as nx
import math


__all__ = ["convert_to_discriminating_cotree", "create_tree", "construct_random_cograph"]


def convert_to_discriminating_cotree(tree: nx.DiGraph, root=None):
    if root is None:
        for node in tree.nodes:
            if tree.in_degree(node) == 0:
                root = node

    root_data = tree.nodes[root]
    if root_data['symbol'] not in ['e', 'b', 'u']:
        return

    for child in list(tree.successors(root)):
        convert_to_discriminating_cotree(tree, child)
        child_data = tree.nodes[child]
        if root_data['symbol'] == child_data['symbol']:
            for sub_child in tree.successors(child):
                tree.add_edge(root, sub_child)
            tree.remove_node(child)


def create_tree(nodes: List[Any]):

    tree = nx.DiGraph()
    _create_sub_tree(tree, nodes, 0, None, None, options=['e', 'b', 'u'])

    return tree


def _create_sub_tree(tree, partition, next_idx, parent_idx, parent_option, options: list):
    if len(partition) == 1:
        tree.add_node(next_idx, symbol=partition[0])
        tree.add_edge(parent_idx, next_idx)
        return next_idx
    else:
        # half the partition set
        half = random.choice(list(range(1, len(partition))))
        sub_partition1 = partition[:half]
        sub_partition2 = partition[half:]

        # add parent node with random option
        option = random.sample(options, 1)[0]
        if parent_option is None or parent_option != option:
            tree.add_node(next_idx, symbol=option)
            if parent_idx is not None:
                tree.add_edge(parent_idx, next_idx)
            reference_idx = next_idx

            if option == 'e':
                sub_options1 = ['e']
                sub_options2 = ['e']
            elif option == 'u':
                sub_options1 = [x for x in options if x in ['e', 'u']]
                sub_options2 = options
            else:
                sub_options1 = options
                sub_options2 = options

        else:
            reference_idx = parent_idx

            sub_options1 = options
            sub_options2 = options

        last_idx = _create_sub_tree(tree, sub_partition1, next_idx + 1, reference_idx, option, options=sub_options1)
        last_idx = _create_sub_tree(tree, sub_partition2, last_idx + 1, reference_idx, option, options=sub_options2)
        return last_idx


def construct_random_cograph(nodes):
    tree = nx.DiGraph()

    sub_trees = []
    for i, x in enumerate(nodes):
        tree.add_node(i, symbol=x)
        sub_trees.append([i])

    while len(sub_trees) > 1:
        idx1, idx2 = random.sample(sub_trees, 2)

        option = random.choice(['e', 'b', 'u'])

        if option in ['u', 'b']:
            for a in idx1:
                for b in idx2:
                    tree.add_edge(a, b)
                    if option == 'b':
                        tree.add_edge(b, a)

        idx1 += idx2
        sub_trees.remove(idx2)

    return tree
