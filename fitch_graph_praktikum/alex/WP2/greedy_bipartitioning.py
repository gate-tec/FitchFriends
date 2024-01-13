import random
import numpy
import copy
from fitch_graph_praktikum.alex.WP2.weight_scoring_for_partitioning import average_weight_scoring, sum_weight_scoring
from fitch_graph_praktikum.alex.WP2.full_weighted_relations import generate_full_weighted_relations
from fitch_graph_praktikum.util.lib import partition_heuristic_scaffold
import networkx as nx


def greedy_bipartition_sum(graph: nx.Graph):
    # def update_edges_cut(nodes: list):
    #     cutted_edges = dict()
    #     for n0 in nodes[0]:
    #         for n1 in nodes[1]:
    #             if (n0, n1) or (n1, n0) in edges.keys():
    #                 k = (n0, n1) if (n0, n1) in edges.keys() else (n1, n0)
    #                 v = edges[k]
    #                 cutted_edges.update({k: v})
    #                 edges_to_cut.pop(k)
    #     return cutted_edges

    # def calc_score_sum(edges_cut: dict):
    #     score = sum(edges_cut[e] for e in edges_cut)
    #     return score if len(edges_cut) > 0 else 0
    #
    # def calc_score_ave(edges_cut: dict):
    #     score = sum(edges_cut[e] for e in edges_cut) / len(edges_cut) if len(edges_cut) > 0 else 0
    #     return score if len(edges_cut) > 0 else 0

    def return_neighbours_new_part(node_list):
        neighbours = set()
        for n in node_list[1]:
            neighbours.update({neighbour for neighbour in list(graph.neighbors(n))})
        for n in node_list[1]:
            try:
                neighbours.remove(n)
            except(KeyError):
                continue
        return neighbours

    def best_greedy_move(nodelist: [], reference_score: float = None):
        """to bipartition the graphs nodes, we check first, whether the selected node lies in the neighbourhood of nodelist[1].
        This is logically correct to avoid partitions into more than two communities. Also it would never appear for nodelist[0]
        to contain no neighbors of nodelist[1] - since this would mean, all nodes would have been moved to nodelist[1]
        - which would worsen the score to zero again"""
        neighbours = return_neighbours_new_part(nodelist)
        nodes_updated = False

        for n0 in nodelist[0]:
            if n0 not in neighbours:
                # useful, since graph is full-connected and at least one element
                # always is in nodelist[0] adjacent to one node of nodelist[1]
                continue
            else:
                greedy_test = copy.deepcopy(nodelist)
                greedy_test[1].append(n0)
                greedy_test[0].remove(n0)
                greedy_test_score = sum_weight_scoring(greedy_test[0], greedy_test[1], edges)

                if reference_score is None:
                    # update reference score.
                    reference_score = greedy_test_score-1
                if greedy_test_score > reference_score:
                    # update reference score.
                    reference_score = greedy_test_score
                    nodes_updated = greedy_test

        if nodes_updated is not False:
            return nodes_updated
        else:
            return False

    def adjust_greedy(nodelist: [], reference_score: float):
        """re-partitions one node from nodelist[1] back to nodelist[0] except for the last node
        and based on  this reperforms the best greedy move"""

        greedy_adjusted = False

        index_last_node_to_move = len(nodelist[1]) - 1
        # probiere alle elemente bis auf das letzte aus:
        for i in range(index_last_node_to_move):
            nodes_to_rollback = copy.deepcopy(nodelist)
            node_to_move = nodes_to_rollback[1][i]
            nodes_to_rollback[0].append(node_to_move)
            nodes_to_rollback[1].remove(node_to_move)

            nodes_adjusted = best_greedy_move(nodes_to_rollback, reference_score=reference_score)
            if nodes_adjusted is not False:
                score_adjusted = sum_weight_scoring(nodes_adjusted[0], nodes_adjusted[1], edges)
                reference_score = score_adjusted
                greedy_adjusted = nodes_adjusted

        return greedy_adjusted

    # cut = max(edges_to_cut.keys(), key=edges_to_cut.get)
    # cut_node = cut[0]
    # nodes[1].append(nodes[0].pop(cut_node))
    #
    # edges_cut = update_edges_cut(nodes)
    # score = calc_score_sum()

    ###################################################################################################################
    nodes = [list(graph.nodes), []]
    edges = {(n1, n2): v['weight'] for n1, n2, v in graph.edges(data=True)}
    edges.update({(n2, n1): v['weight'] for n1, n2, v in graph.edges(data=True)})

    starter = random.choice(nodes[0])
    nodes[0].remove(starter)
    nodes[1].append(starter)


    score = sum_weight_scoring(nodes[0], nodes[1], edges)
    print("nodes:", nodes, ", initial score:", score)

    score_old = score - 1
    while score > score_old:
        score_old = score

        nodes_updated = best_greedy_move(nodes, reference_score=score_old)
        if nodes_updated is False:
            nodes_updated = adjust_greedy(nodes, score_old)

        if nodes_updated is False:
            continue

        else:
            nodes = nodes_updated
            score = sum_weight_scoring(nodes[0], nodes[1], edges)

    print("nodes:", nodes, ", initial score:", score)
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
