import random
from copy import deepcopy
import networkx as nx
import statistics
from fitch_graph_praktikum.util.helper_functions import *

def graph_to_rel(graph: nx.DiGraph):
    nodes = list(graph.nodes())

    # We remap the nodes here so the graph always has nodes 0...n.
    remap = {nodes[k]: k for k in range(0, len(nodes))}
    graph = nx.relabel_nodes(graph, remap)

    nodes = [x for x in range(0, len(nodes))]

    edges = list(graph.edges())
    n = len(nodes)

    relations = {1: [], 0: [], "d": []}

    # Extract the corresponding relation type for each tuple of nodes.
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
    # Init graph and nodes.
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)

    # Add all the edges.
    graph.add_edges_from(relations[1])
    graph.add_edges_from(relations["d"])

    return graph


def check_fitch_graph(graph: nx.DiGraph):
    nodes = graph.nodes

    # In O(n^3) we check every combination of nodes for forbidden subgraphs. Each if block corresponds
    # to a forbidden subgraph (See Figure in script)
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
                elif (x, y) not in edges and (y, x) in edges and \
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
    #
    # You need a special structure, in case you want to use this to parse randomly generated cotrees.
    # Each node in the cotree needs to have a label "symbol". Inner Nodes need the "symbol" "u", "b", or "e"
    # where u encodes ->1 nodes, b encodes 1 nodes, and e encodes 0 nodes.
    # Leaves are expexted to have an integer value of 0...n where n is the total number of nodes
    # in the underlying graph. For ->1 nodes, the order of childen is realized by the node ids, i.e. the integer value
    # used to initiate a node. For example, on an empty DiGraph graph:
    #
    # graph.add_node(0, symbol="u")
    # graph.add_node(1, symbol=0)
    # graph.add_node(2, symbol=1)
    #
    # grap.add_edge(0, 1)
    # graph.add_edge(0, 2)
    #
    # yields the cotree encoding the graph (0) -> (1).

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

    completed_graph = {0: [], 1: [], "d": []}
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
                    for c1 in range(c0 + 1, len(clusters[q]["leaves"])):
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
                    for c1 in range(c0 + 1, len(clusters[q]["order"])):
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

    # symmetric=False leads to different weights for (x,y) and (y,x) for tuples in the relation.
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
    # Implements the symmetric distance as descibed in the practical script.
    difference_empty = set(relations_0[0]).difference(set(relations_1[0])).union(
        set(relations_1[0]).difference(set(relations_0[0])))
    difference_bi = set(relations_0[1]).difference(set(relations_1[1])).union(
        set(relations_1[1]).difference(set(relations_0[1])))
    difference_uni = set(relations_0["d"]).difference(set(relations_1["d"])).union(
        set(relations_1["d"]).difference(set(relations_0["d"])))

    relative_difference = len(difference_empty.union(difference_bi).union(difference_uni))

    # We do not have (n * (n - 1))/2 as we expect a directed graph.
    all_edges = n * (n - 1)

    return relative_difference / all_edges


def partition_heuristic_scaffold(uni_weighted: dict, bi_weighted: dict, empty_weighted: dict, nodes: list,
                                 partition_function, scoring_function, relations=None, uni=True, bi=True, median=False, reciprocal=False):

    # Initialize relations - here we collect the relations at each recursive step
    if relations == None:
        relations = {0: [], 1: [], "d": []}

    # Recursion abort condition
    if len(nodes) == 1:
        return relations

    # Create three auxilliary graphs
    graph_bi = nx.Graph()
    graph_empty = nx.Graph()
    graph_uni = nx.Graph()

    # Compute the median for each of the weights.
    bi_list = [bi_weighted[k] for k in bi_weighted.keys()]
    uni_list = [uni_weighted[k] for k in uni_weighted.keys()]
    empty_list = [empty_weighted[k] for k in empty_weighted.keys()]

    bi_med = statistics.median(bi_list)
    uni_med = statistics.median(uni_list)
    empty_med = statistics.median(empty_list)

    # Add all the edge weights. You could also add a filter here that,
    # for example, could add only edges above a certain weight
    for i in range(0, len(nodes)):
        for j in range(0, len(nodes)):
            if i == j:
                continue
            if median:
                if bi_weighted[(nodes[i], nodes[j])] <= bi_med:
                    # graph_bi.add_edge(nodes[i], nodes[j], weight=1.0)
                    graph_bi.add_edge(nodes[i], nodes[j], weight=bi_weighted[(nodes[i], nodes[j])])
                if empty_weighted[(nodes[i], nodes[j])] <= empty_med:
                    graph_empty.add_edge(nodes[i], nodes[j],
                                           weight=empty_weighted[(nodes[i], nodes[j])])
                if (uni_weighted[(nodes[i], nodes[j])] + uni_weighted[(nodes[j], nodes[i])])/2 <= uni_med:
                    graph_uni.add_edge(nodes[i], nodes[j],
                                         weight=(uni_weighted[(nodes[i], nodes[j])] + uni_weighted[(nodes[j], nodes[i])])/2)
            else:
                if reciprocal:
                    graph_bi.add_edge(nodes[i], nodes[j], weight=1/bi_weighted[(nodes[i], nodes[j])])
                    graph_empty.add_edge(nodes[i], nodes[j],
                                         weight=1/empty_weighted[(nodes[i], nodes[j])])
                    graph_uni.add_edge(nodes[i], nodes[j],
                                       weight=1/((uni_weighted[(nodes[i], nodes[j])] + uni_weighted[
                                           (nodes[j], nodes[i])]) / 2))
                else:
                    graph_bi.add_edge(nodes[i], nodes[j], weight=bi_weighted[(nodes[i], nodes[j])])
                    graph_empty.add_edge(nodes[i], nodes[j],
                                           weight=empty_weighted[(nodes[i], nodes[j])])
                    graph_uni.add_edge(nodes[i], nodes[j],
                                             weight=((uni_weighted[(nodes[i], nodes[j])] + uni_weighted[(nodes[j], nodes[i])])/2))


    # If only two nodes are left, we can only choose one partition.
    if len(nodes) == 2:
        uni_partition = [[nodes[0]], [nodes[1]]]
        bi_partition = [[nodes[0]], [nodes[1]]]
        empty_partition = [[nodes[0]], [nodes[1]]]

    # Otherwise we partition as usual. Here it is expected that the partition functions returns a list of lists,
    # for example [[0, 1], [2, 3]]
    else:
        bi_partition = partition_function(graph_bi)
        uni_partition = partition_function(graph_uni)
        empty_partition = partition_function(graph_empty)

    # Setting up variables for finding the scores of each partition.
    score_bi = 0
    score_empty = 0
    score_uni = 0

    left_bi, right_bi = [], []
    left_uni, right_uni = [], []
    left_empty, right_empty = [], []

    # This part is set up such that it can also deal with partitions that are not bipartitions of the vertex set
    for l in range(len(bi_partition)):

        # Choose one partition set as 'left'
        left = list(bi_partition[l])
        right = []

        # All the other partition sets are added to 'right'
        for r in range(0, len(bi_partition)):
            if r == l:
                continue
            right += list(bi_partition[r])

        # Score the current partition
        score = scoring_function(left, right, bi_weighted)

        # Compare it to the last and currently highest scoring partition
        if score > score_bi or (left_bi == [] and right_bi == []):
            score_bi = score
            left_bi = left
            right_bi = right

    for l in range(len(empty_partition)):

        # Choose one partition set as 'left'
        left = list(empty_partition[l])
        right = []

        # All the other partition sets are added to 'right'
        for r in range(0, len(empty_partition)):
            if r == l:
                continue
            right += list(empty_partition[r])

        # Score the current partition
        score = scoring_function(left, right, empty_weighted)

        # Compare it to the last and currently highest scoring partition
        if score > score_empty or (left_empty == [] and right_empty == []):
            score_empty = score
            left_empty = left
            right_empty = right

    for l in range(len(uni_partition)):

        # Choose one partition set as 'left'
        left = list(uni_partition[l])
        right = []

        # All the other partition sets are added to 'right'
        for r in range(0, len(uni_partition)):
            if r == l:
                continue
            right += list(uni_partition[r])

        # Score the current partition
        score = scoring_function(left, right, uni_weighted)

        # Compare it to the last and currently highest scoring partition
        if score > score_uni or (left_uni == [] and right_uni == []):
            score_uni = score
            left_uni = left
            right_uni = right

        # Score the other direction
        score = scoring_function(right, left, uni_weighted)
        if score > score_uni or (left_uni == [] and right_uni == []):
            score_uni = score
            left_uni = right
            right_uni = left

    # Some booleans for the partition we choose
    part_empty = False
    part_uni = False
    part_bi = False

    # If bidirectional partition yields highest score and we are allowed to
    # introduce an inner '1' node in the Fitch cotree, we proceed
    if bi and score_bi >= score_empty and score_bi >= score_uni:
        relations[1] += [(i, j) for i in right_bi for j in left_bi]
        relations[1] += [(j, i) for i in right_bi for j in left_bi]
        part_bi = True

    # If unidirectional partition yields highest score and we are allowed to
    # introduce an inner '->1' node in the Fitch cotree, we proceed
    if uni and score_uni >= score_empty and not part_bi:
        relations["d"] += [(i, j) for i in left_uni for j in right_uni]
        part_uni = True

    # If empty partition yields highest score, or we are not allowed to
    # introduce a '1' or '->1' in the Fitch cotree
    if not part_bi and not part_uni:
        relations[0] += [(i, j) for i in right_empty for j in left_empty]
        relations[0] += [(j, i) for i in right_empty for j in left_empty]
        part_empty = True

    # Debug lines. Remove the # to output the vertex set, partitions, and scores at each recursive step.
    # print("VERTEX SET  ", nodes)
    # print("UNI SCORE   ",  score_uni, ", PARTITION ", left_uni, " ", right_uni)
    # print("BI SCORE    ",  score_bi, ", PARTITION ", left_bi, " ", right_bi)
    # print("EMPTY SCORE ",  score_empty, ", PARTITION ", left_empty, " ", right_empty)
    # print("--------------------")
    # input()

    if part_bi:
        # Continue to recursively partition left_bi and right_bi. Resulting edges are collected in 'relations'.
        relations = partition_heuristic_scaffold(uni_weighted, bi_weighted, empty_weighted, left_bi, partition_function,
                                                 scoring_function, relations, uni=uni, bi=bi)
        relations = partition_heuristic_scaffold(uni_weighted, bi_weighted, empty_weighted, right_bi,
                                                 partition_function,
                                                 scoring_function, relations)
        return relations

    if part_empty:
        # Continue to recursively partition left_empty and right_empty. Resulting edges are collected in 'relations'.
        # We forbid to partition G_1 and G_->1 for left_empty and right_empty.
        relations = partition_heuristic_scaffold(uni_weighted, bi_weighted, empty_weighted, left_empty,
                                                 partition_function,
                                                 scoring_function, relations, uni=False, bi=False)
        relations = partition_heuristic_scaffold(uni_weighted, bi_weighted, empty_weighted, right_empty,
                                                 partition_function,
                                                 scoring_function, relations, uni=False, bi=False)
        return relations

    if part_uni:
        # Continue to recursively partition left_uni and right_uni. Resulting edges are collected in 'relations'.
        # We forbid to partition G_1 and for left_uni.
        relations = partition_heuristic_scaffold(uni_weighted, bi_weighted, empty_weighted, left_uni,
                                                 partition_function,
                                                 scoring_function, relations, uni=uni, bi=False)
        relations = partition_heuristic_scaffold(uni_weighted, bi_weighted, empty_weighted, right_uni,
                                                 partition_function,
                                                 scoring_function, relations, uni=uni, bi=bi)
        return relations


def algorithm_one(relations, nodes, order, symbol_attr='symbol'):
    """
    Constructs a fitch di-cotree T that explains
    the partial set (E_0, E_1, E_d) on the set of nodes.

    Input:
    - nodes : graph nodes.
    - relations : relation dict containing partial set.
    - symbol_attr : Node attribute for inner-node symbol and leaf label, only important for cotree_to_rel function

    Output:
    - T : networkx.DiGraph
          Represents the cotree explaining the partial set described by relations.
          The tree has its root at node 0.

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
    w_relations_sorted = sorted(w_relations_new, reverse=True, key=lambda x: max([x[1][k] for k in x[1].keys()]))

    for t in w_relations_sorted:
        x, y = tuple(t[0])[0], tuple(t[0])[1]

        Rs = sorted(t[1], key=lambda x: t[1][x], reverse=True)
        # print(t[1])
        # print(Rs)
        # print("-----")
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
                algorithm_one({0: E_aux[0], 1: E_aux[1], "d": E_aux['d']}, V, (0, 1, 2))
                E_star = E_aux
                flag = False
                break
            except NotFitchSatError as e:
                pass
        if flag:
            raise NoSatRelation(f'Can not find a satisfiable relation for {(x, y)}')

    return {0: list(E_star[0]), 1: list(E_star[1]), "d": list(E_star["d"])}


if __name__ == '__main__':
    # Init some nodes
    nodes = [0, 1, 2]

    # Init some partial relations
    relation = {
        0: [],
        1: [(0, 1), (1, 0)],
        "d": [(1, 2)]
    }

    # Init weights for unidirectional relations, bidirectional relations, and empty relations
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

    # Compute a Cotree using the partial set defined prior.
    fitch_cotree_210 = algorithm_one(relation, nodes, (1, 2, 0))

    # Compute a Cotree using the partial set defined prior but with a different order of Rules.
    fitch_cotree_012 = algorithm_one(relation, nodes, (0, 1, 2))

    # Parse the cotrees just computed into a relations dictionary.
    fitch_relations_210 = cotree_to_rel(fitch_cotree_210)
    fitch_relations_012 = cotree_to_rel(fitch_cotree_012)

    # Run the greedy algorithm on the weighted relations initialized above.
    fitch_relations_greedy = algorithm_two(nodes, uni_weighted, bi_weighted, empty_weighted)

    # Generate weights for "1" with the random.uniform sampling between 1.0 and 1.5
    test_weights_bi = generate_weights(relation[1], random.uniform, [1.0, 1.5], symmetric=True)

    # Generate weights for "d" with the random.random generator. As it takes no arguments, the parameters are empty.
    test_weights_uni = generate_weights(relation["d"], random.random, [], symmetric=False)

    # Check a if the graph reconstructed from fitch_relations_210 is fitch-sat
    fitch_graph_210 = rel_to_fitch(fitch_relations_210, nodes)
    is_fitch = check_fitch_graph(fitch_graph_210)

    # Initialize an empty cotree.
    cotree = nx.DiGraph()

    # Add one inner node ->1 with three leaves 0, 1, and 2. See the function cotree_to_rel for more
    # information on how nodes/edges need to be defined.
    cotree.add_node(0, symbol="u")
    cotree.add_node(1, symbol=0)
    cotree.add_node(2, symbol=1)
    cotree.add_node(3, symbol=2)

    cotree.add_edge(0, 1)
    cotree.add_edge(0, 2)
    cotree.add_edge(0, 3)

    # Translate cotree to relations
    decoded_cotree = cotree_to_rel(cotree)

    # Thats how you can call the partition heuristic, simp_part and simp_score still need to be implemented.
    # fitch_relations_partition = partition_heuristic_scaffold(uni_weighted, bi_weighted, empty_weighted, nodes, simp_part, simp_score)

    # Output
    print("Input relations           - ", relation)
    print("Completed E*              - ", fitch_relations_210)
    print("Is Fitch Sat?             - ", is_fitch)
    print("Different Completed E*    - ", fitch_relations_012)
    print("Greedy Completed E*       - ", fitch_relations_greedy)
    print("Generated weights 1       - ", test_weights_bi)
    print("Generated weights d       - ", test_weights_uni)
    print("Extracted cotree rels     - ", decoded_cotree)
    # print("Partition Completed E*    - ", fitch_relations_partition)
