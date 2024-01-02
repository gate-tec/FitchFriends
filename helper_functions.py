import random
from copy import deepcopy
import networkx as nx

# This file just contains some helper functions for Algorithm 1 and Algorithm 2.

def evaluate_conditions(V, E_sets, order):
    """
    This function returns a label for the node "x" in the cotree representing V and E_sets.
    In addition, it returns a partition of V for the children of "x".
    """
    if len(V) == 1:
        return [], V.pop()
    res = -1

    for o in order:
        if o == 0:
            res = check_cond_0(V, E_sets)
        elif o == 1:
            res = check_cond_1(V, E_sets)
        else:
            res = check_cond_d(V, E_sets)

        if res != -1:
            return res

    if res == -1:
        raise NotFitchSatError("DIDNT WORK!")

def check_cond_0(V, E_sets):
    G_0 = get_G_i(0, V, E_sets)
    if is_edge_less(G_0):
        return [{x} for x in G_0], "e"
    return -1

def check_cond_1(V, E_sets):
    G_1 = get_G_i(1, V, E_sets)
    CCs_1 = list(nx.weakly_connected_components(G_1))
    if len(CCs_1) > 1:
        return CCs_1, "b"
    return -1

def check_cond_d(V, E_sets):
    G_0 = get_G_i(0, V, E_sets)
    G_d = get_G_i('d', V, E_sets)
    CCs_d = list(map(frozenset, nx.strongly_connected_components(G_d)))
    if len(CCs_d) > 1:
        I_set = set()
        for CC in CCs_d:
            if is_edge_less(nx.induced_subgraph(G_0, CC)):
                I_set.add(CC)
        if len(I_set) == 0:
            return -1
        quotient = nx.quotient_graph(G_d, CCs_d)
        if not nx.is_directed_acyclic_graph(quotient):
            return -1
        I_set = I_set.intersection({X for X in quotient if quotient.in_degree[X] == 0})
        if len(I_set) == 0:
            return -1
        CC_star = I_set.pop()
        return list(map(set, (CC_star, V - CC_star))), 'u'
    return -1

def get_G_i(i, V: set, E_sets: tuple):
    # Select edge set
    if i == 0:
        E = E_sets[1].union(E_sets[2])
    elif i == 1:
        E = E_sets[0].union(E_sets[2])
    elif i == 'd':
        E = E_sets[0].union(E_sets[1]).union(E_sets[2])
    else:
        raise ValueError('wrong value for "i"')

    # Filter edges by nodes
    E = filter_E(V, E)
    # Create graph
    G = nx.DiGraph()
    G.add_nodes_from(V)
    G.add_edges_from(E)
    return G

def is_edge_less(G):
    return len(G.edges) == 0

def filter_E(V, E):
    return tuple((x, y) for x, y in E if V.issuperset({x, y}))

class NoSatRelation(Exception):
    """Raised when in greedy algorithm there is not a satisfiable relation for two nodes"""
    pass

class NotFitchSatError(Exception):
    """Raised when a partial set is not fitch satisfiable"""
    pass

def classify_rel(X):
    if type(X) == tuple:
        return 'd'
    if X == 'bidirectional':
        return 1
    return 0
