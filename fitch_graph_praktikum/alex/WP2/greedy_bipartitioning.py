import random
import numpy
from fitch_graph_praktikum.alex.WP2.full_weighted_relations import generate_full_weighted_relations
from fitch_graph_praktikum.util.lib import partition_heuristic_scaffold
import networkx as nx


def greedy_bipartition_sum(graph: nx.Graph):

    def update_edges_cut(nodes: list):
        cutted_edges = dict()
        for n0 in nodes[0]:
            for n1 in nodes[1]:
                if (n0, n1) in edges:
                    edges_cut.update(edges_to_cut.pop(n0, n1))
        return cutted_edges

    def calc_score_sum(edges_cut: dict):
        score = sum(edges_cut[e] for e in edges_cut)
        return score

    def calc_score_ave(edges_cut: dict):
        score = sum(edges_cut[e] for e in edges_cut) / len(edges_cut) if len(edges_cut) > 0 else 0
        return score

    def return_neighbours_new_part(node_list):
        neighbours = set()
        for n in node_list[1]:
            neighbours.update({neighbour for neighbour in list(graph.neighbors(n))})
        for n in node_list[1]:
            neighbours.remove(n)

        return neighbours

    def best_greedy_move(nodeslist: [], reference_score: float = None):
        neighbours = return_neighbours_new_part(nodeslist)
        nodes_updated = False
        greedy_prime = nodeslist

        for n0 in nodeslist[0]:
            if n0 not in neighbours:
                continue
            else:
                updated = True
                greedy_nodes = nodeslist
                greedy_nodes[1].append(greedy_nodes[0].pop(n0))
                greedy_test_score = calc_score_sum(update_edges_cut(greedy_nodes))

                if greedy_test_score > reference_score or reference_score is None:
                    greedy_score = greedy_test_score
                    greedy_prime = greedy_nodes
                    if reference_score is None:
                        reference_score = greedy_score

        if nodes_updated is True:
            return greedy_prime
        else:
            return False

    ####################################################################################################################
    nodes = [list(graph.nodes), []]
    edges = {(n1, n2): v['weight'] for n1, n2, v in graph.edges(data=True)}
    edges_to_cut = edges
    edges_cut = dict()

    # cut = max(edges_to_cut.keys(), key=edges_to_cut.get)
    # cut_node = cut[0]
    # nodes[1].append(nodes[0].pop(cut_node))
    #
    # edges_cut = update_edges_cut(nodes)
    # score = calc_score_sum()

    nodes = best_greedy_move(nodes)
    edges_cut = update_edges_cut(nodes)
    score = calc_score_sum(edges_cut)
    print("nodes, ", nodes, ", score_new :", score)

    score_old = score - 1
    while score > score_old:
        score_old = score

        greedy_change = best_greedy_move(nodes, reference_score=score_old)

        if greedy_change is not False:
            nodes = greedy_change
            edges_cut = update_edges_cut(nodes)
            score = calc_score_ave(edges_cut)
            print("nodes, ", nodes, ", score_new :", score)

        else:
            greedy_prime_score = score_old
            # greedy next move fehlgeschlagen -> ein Element aus nodes[1] zurück zu nodes[0] packen
            for i in range(len(nodes[1]) - 1):
                greedy_nodes = nodes
                greedy_nodes[0].append(greedy_nodes[1].pop(greedy_nodes[1][i]))

                greedy_change = best_greedy_move(nodes, reference_score=score_old)

                if greedy_change is not False:
                    greedy_score =  calc_score_sum(update_edges_cut(greedy_change))
                    if greedy_score <= greedy_prime_score:
                        continue
                    else:
                        greedy_prime = greedy_change
                        greedy_prime_score = greedy_score
            score = greedy_prime_score

    return nodes








if __name__ == '__main__':
    # print(numpy.random.normal)
    # test_relations = {0: [(1, 2), (2, 1)], 1: [(0, 1), (1, 0)], 'd': [(0, 2)]}
    # test_weights_sym = generate_full_weighted_relations(3, test_relations, numpy.random.normal, [3, 0.75],
    #                                                     numpy.random.normal, [1, 0.5], symmetric=True)
    # test_weights_asym = generate_weighted_relations(test_relations, numpy.random.normal, [3, 0.75], symmetric=False)
    # print(test_weights_sym)
    # for k, v in test_weights_sym.items():
    #     print(f"{k}: {v}")
    #
    # test = partition_heuristic_scaffold({}, {}, {}, [], partition_function=0, scoring_function=0)

    edges = [(0, 1, 2.0), (2, 0, 3.0), (1, 2, 0.5), (0, 3, 0.9), (1, 3, 1.5), (2, 3, 2.0)]
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    print(greedy_bipartition_sum(G))
