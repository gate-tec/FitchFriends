from copy import copy, deepcopy
import networkx as nx
import matplotlib.pyplot as plt

def graph_to_rel(graph: nx.DiGraph):
    nodes = list(graph.nodes())
    remap = {nodes[k]: k for k in range(0, len(nodes))}
    graph = nx.relabel_nodes(graph, remap)

    nodes = [x for x in range(0, len(nodes))]

    edges = list(graph.edges())
    n = len(nodes)

    relations = {1: [], 0: [], "d": []}

    for x in range(0, len(nodes)):
        for y in range(0, len(nodes)):
            if x == y:
                continue
            if (x, y) in edges and (y, x) in edges:
                relations[1] += [(x, y)]
            elif (x, y) in edges and (y, x) not in edges:
                relations["d"] += [(x, y)]
            elif (x, y) not in edges and (y, x) not in edges:
                relations[0] += [(x, y)]

    return relations

def rel_to_fitch(relations: dict, nodes):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)

    graph.add_edges_from(relations[1])
    graph.add_edges_from(relations["d"])

    return graph

def check_fitch_graph(graph: nx.DiGraph):
    nodes = graph.nodes

    for x in nodes:
        for y in nodes:
            for z in nodes:
                if x == y or x == z or y == z:
                    continue
                edges = nx.induced_subgraph(graph, [x, y, z]).edges()

                # F1
                if (x, y) in edges and (y, x) not in edges \
                        and (x, z) not in edges and (z, x) not in edges \
                        and (y, z) not in edges and (z, y) not in edges:
                    return False

                # F2
                elif (x, y) not in edges and (y, x) in edges and\
                        (x, z) in edges and (z, x) not in edges \
                        and (y, z) not in edges and (z, y) in edges:
                    return False

                # F3
                elif (x, y) not in edges and (y, x) in edges \
                        and (x, z) not in edges and (z, x) not in edges \
                        and (y, z) not in edges and (z, y) in edges:
                    return False

                # F4
                elif (x, y) not in edges and (y, x) in edges \
                        and (x, z) in edges and (z, x) in edges \
                        and (y, z) not in edges and (z, y) in edges:
                    return False

                # F5
                elif (x, y) in edges and (y, x) in edges \
                        and (x, z) not in edges and (z, x) not in edges \
                        and (y, z) not in edges and (z, y) not in edges:
                    return False

                # F6
                elif (x, y) in edges and (y, x) in edges \
                        and (x, z) not in edges and (z, x) not in edges \
                        and (y, z) not in edges and (z, y) in edges:
                    return False

                # F7
                elif (x, y) in edges and (y, x) in edges \
                        and (x, z) not in edges and (z, x) not in edges \
                        and (y, z) in edges and (z, y) not in edges:
                    return False

                # F8
                elif (x, y) in edges and (y, x) not in edges \
                        and (x, z) in edges and (z, x) in edges \
                        and (y, z) not in edges and (z, y) in edges:
                    return False

    return True

def cotree_to_rel(cotree):

    ct = cotree
    clusters = {}
    node_attributes = nx.get_node_attributes(ct, "symbol")

    for n in ct.nodes(data=True):
        if not isinstance(n[1]["symbol"], int):
            successors = nx.bfs_tree(ct, n[0])
            clusters[n[0]] = {"direct_leaves": [],
                              "leaves": [], "clustertype": n[1]["symbol"],
                              'successor_clusters': [node for node in ct[n[0]] if
                                                     node_attributes[node] in ['u', 'b', 'e']], "order": []}
            for s in successors:
                if node_attributes[s] in ["u", "b", "e"]:
                    continue
                else:
                    clusters[n[0]]["leaves"] += [s]
        else:
            clusters[list(ct.predecessors(n[0]))[0]]["direct_leaves"] += [n[0]]


    completed_graph = {0:[], 1:[], "d":[]}
    count_uni_edges = 0

    queue = [k for k in clusters if clusters[k]['successor_clusters'] == []]
    while len(queue) != 0:
        q = queue.pop(0)

        if clusters[q]["order"] == []:
            if clusters[q]["clustertype"] == "b":
                for x in clusters[q]["leaves"]:
                    for y in clusters[q]["leaves"]:
                        if x == y:
                            continue
                        else:
                            completed_graph[1] += [(x, y)]
            elif clusters[q]["clustertype"] == "e":
                for x in clusters[q]["leaves"]:
                    for y in clusters[q]["leaves"]:
                        if x == y:
                            continue
                        else:
                            completed_graph[0] += [(x, y)]
            elif clusters[q]["clustertype"] == "u":
                for c0 in range(len(clusters[q]["leaves"])):
                    x = clusters[q]["leaves"][c0]
                    for c1 in range(c0+1, len(clusters[q]["leaves"])):
                        y = clusters[q]["leaves"][c1]
                        count_uni_edges += 1
                        completed_graph["d"] += [(x, y)]

        else:
            if len(clusters[q]["direct_leaves"]) != 0:
                for l in clusters[q]["direct_leaves"]:
                    clusters[q]["order"] += [(l, [l])]

            clusters[q]["order"] = sorted(clusters[q]["order"], key=lambda x: x[0])
            for c in range(0, len(clusters[q]["order"])):
                clusters[q]["order"][c] = clusters[q]["order"][c][1]

            if clusters[q]["clustertype"] == "b":
                for c0 in range(len(clusters[q]["order"])):
                    for c1 in range(len(clusters[q]["order"])):
                        if c0 == c1:
                            continue
                        else:
                            cluster_0 = clusters[q]["order"][c0]
                            cluster_1 = clusters[q]["order"][c1]
                            for x in cluster_0:
                                for y in cluster_1:
                                    completed_graph[1] += [(x, y)]

            elif clusters[q]["clustertype"] == "e":
                for c0 in range(len(clusters[q]["order"])):
                    for c1 in range(len(clusters[q]["order"])):
                        if c0 == c1:
                            continue
                        else:
                            cluster_0 = clusters[q]["order"][c0]
                            cluster_1 = clusters[q]["order"][c1]
                            for x in cluster_0:
                                for y in cluster_1:
                                    completed_graph[0] += [(x, y)]

            elif clusters[q]["clustertype"] == "u":
                for c0 in range(len(clusters[q]["order"])):
                    cluster_0 = clusters[q]["order"][c0]
                    for c1 in range(c0+1, len(clusters[q]["order"])):
                        cluster_1 = clusters[q]["order"][c1]
                        for x in cluster_0:
                            for y in cluster_1:
                                count_uni_edges += 1
                                completed_graph["d"] += [(x, y)]

        if len(clusters) != 1:
            insert_cluster = [cl for cl in clusters.keys() if q in clusters[cl]["successor_clusters"]][0]
            clusters[insert_cluster]["order"] += [(q, clusters[q]["leaves"])]
            clusters[insert_cluster]["successor_clusters"].pop(clusters[insert_cluster]["successor_clusters"].index(q))

        clusters.pop(q)

        if len(queue) == 0:
            queue = [k for k in clusters if clusters[k]['successor_clusters'] == []]

    remapped_graph = {0: [], 1: [], "d": []}

    for k in completed_graph:
        for e in completed_graph[k]:
            remapped_graph[k] += [(node_attributes[e[0]], node_attributes[e[1]])]

    completed_graph = remapped_graph

    return completed_graph

def generate_weights(relation, distribution, parameters, symmetric=True):
    weighted_relation = {}
    if not symmetric:
        weighted_relation = {r: distribution(*parameters) for r in relation}
        return weighted_relation
    else:
        for r in relation:
            weight = distribution(*parameters)
            weighted_relation[r] = weight
            weighted_relation[(r[1], r[0])] = weight
        return weighted_relation

def sym_diff(relations_0, relations_1, n):

    difference_empty = set(relations_0[0]).difference(set(relations_1[0])).union(
        set(relations_1[0]).difference(set(relations_0[0])))
    difference_bi = set(relations_0[1]).difference(set(relations_1[1])).union(
        set(relations_1[1]).difference(set(relations_0[1])))
    difference_uni = set(relations_0["d"]).difference(set(relations_1["d"])).union(
        set(relations_1["d"]).difference(set(relations_0["d"])))

    relative_difference = len(difference_empty.union(difference_bi).union(difference_uni))

    all_edges = n * (n - 1)

    return relative_difference / all_edges

def partition_heuristic_scaffold(weighted_relations: dict, nodes: list, partition_function, weights=False,
                                 uni=True, bi=True):
    pass

def algorithm_one(relations, nodes, order, symbol_attr='symbol'):
    """
    Constructs a fitch di-cotree T that explains
    the partial set (E_0, E_1, E_d) on the set of nodes V.

    Input:
    - V : set of genes
    - E_0 : set of tuples
            Symmetric relations indicating "no edge"
    - E_1 : set of tuples
            Symmetric relation "x <--> y"
    - E_d : set of tuples
            Non symmetric relation "x --> y"
    - symbol_attr : Node attribute for inner-node symbol and lefs label,

    Output:
    - T : networkx.DiGraph
          Represents the cotree explaining the partial set conformed by E_0, E_1, and E_d.
          The tree have root at node 0.

    Error:
    - NoFitchSat: Raised whebn the input partial set (E_0, E_1, E_d) is not fitch satisfiable.

    """
    E_0 = set(relations[0])
    E_1 = set(relations[1])
    E_d = set(relations["d"])
    V = set(nodes)

    T = nx.DiGraph()
    idx = 1

    # This queue substitutes the recursive part of the algorithm
    queue = [(V, (E_0, E_1, E_d), 0, None)]  # Vertex set, relations set, node name for iteration, dad

    while (len(queue) > 0):
        V, E_sets, n_idx, dad_idx = queue.pop(0)
        # Evaluate conditions
        V_partition, symbol = evaluate_conditions(V, E_sets, order)
        # Create node for iteration
        T.add_node(n_idx, **{symbol_attr: symbol})
        if dad_idx != None:
            T.add_edge(dad_idx, n_idx)
        # Add children to queue
        for V_p in V_partition:
            queue += [(V_p, E_sets, idx, n_idx)]
            idx += 1
    # End
    return T

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

    # Filter edges by noded
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

def algorithm_two(V, variables_uni, variables_bi, variables_empty):
    """
    Searchs for a set of relationships between members of V
    such that if maximizes the scores w_relations and we obtain
    a fitch sat.

    Input:
    - V : set of nodes
    - w_relations : dictionary of dictionaries
                    - 1st level key: frozenset containing two nodes x,y
                    - 2nd level key: a relationship between x & y:
                                     - (x,y) : x points to y
                                     - (y,x) : y points to x
                                     - "bidirectional" : x points to y and y points to x
                                     - None : There are no edges between x and y
                    - (2nd level) value: A score for the specified relation-

    Output:
    - E_star : A 3-tuple of sets of 2-tuples
               Represents a full set (E_0, E_1, E_d).

    Error:
    - NoSatRelation : Raised when we can not find a full set.

    """
    w_relations = {}

    for x in range(0, len(V)):
        for y in range(x + 1, len(V)):
            w_relations[frozenset([x, y])] = {}
            w_relations[frozenset([x, y])][(x, y)] = variables_uni[(x, y)]
            w_relations[frozenset([x, y])][(y, x)] = variables_uni[(y, x)]
            w_relations[frozenset([x, y])]["bidirectional"] = variables_bi[(x, y)]
            w_relations[frozenset([x, y])][None] = variables_empty[(x, y)]

    E_star = {0: set(), 1: set(), 'd': set()}

    w_relations_new = [[k, w_relations[k]] for k in w_relations.keys()]
    w_relations_sorted = sorted(w_relations_new, reverse=True, key=lambda x: max([x[1][k] for k in x[1].keys()]) )

    for t in w_relations_sorted:
        x, y = tuple(t[0])[0], tuple(t[0])[1]

        Rs = sorted(t[1], key= lambda x: t[1][x], reverse=True)
        print(t[1])
        print(Rs)
        print("-----")
        # Find a relation between x and y
        flag = True
        for rel in Rs:
            E_aux = deepcopy(E_star)
            rel_type = classify_rel(rel)
            if rel_type in [0, 1]:
                E_aux[rel_type].update(((x, y), (y, x)))
            elif rel_type == 'd':
                E_aux[rel_type].add(rel)

            try:
                algorithm_one({0:E_aux[0], 1:E_aux[1], "d":E_aux['d']}, V, (0, 1, 2))
                E_star = E_aux
                flag = False
                break
            except NotFitchSatError as e:
                pass
        if flag:
            raise NoSatRelation(f'Can not find a satisfiable relation for {(x, y)}')

    return {0: list(E_star[0]), 1: list(E_star[1]), "d": list(E_star["d"])}

def classify_rel(X):
    if type(X) == tuple:
        return 'd'
    if X == 'bidirectional':
        return 1
    return 0

if __name__ == '__main__':

    nodes = [0, 1, 2]

    relation = {
        0: [],
        1: [(0, 1), (1, 0)],
        "d": [(1, 2)]
    }

    uni_weighted = {
        (1, 2): 100,
        (0, 1): -100,
        (1, 0): -100,
        (2, 1): -100,
        (0, 2): -100,
        (2, 0): -100
    }

    bi_weighted = {
        (0, 1): 100,
        (1, 0): 100,
        (1, 2): -100,
        (2, 1): -100,
        (0, 2): -100,
        (2, 0): -100
    }

    empty_weighted = {
        (0, 1): -100,
        (1, 0): -100,
        (1, 2): -100,
        (2, 1): -100,
        (0, 2): -100,
        (2, 0): -100
    }

    fitch_cotree_210 = algorithm_one(relation, nodes, (1, 2, 0))
    fitch_cotree_012 = algorithm_one(relation, nodes, (0, 1, 2))

    fitch_relations_greedy = algorithm_two(nodes, uni_weighted, bi_weighted, empty_weighted)

    fitch_relations_210 = cotree_to_rel(fitch_cotree_210)
    fitch_relations_012 = cotree_to_rel(fitch_cotree_012)

    print("Input relations           - ", relation)
    print("Completed E*              - ", fitch_relations_210)
    print("Different Completed E*    - ", fitch_relations_012)
    print("Greedy Completed E*       - ", fitch_relations_greedy)

