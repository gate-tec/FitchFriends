from fitch_graph_praktikum.util.lib import algorithm_one, algorithm_two, cotree_to_rel
from fitch_graph_praktikum.nicolas.functions_partial_tuple import convert_to_weighted_relations
from fitch_graph_praktikum.util.typing import RelationDictionary, WeightedRelationDictionary
from fitch_graph_praktikum.util.helper_functions import NoSatRelation, NotFitchSatError

import time


def pipeline_algo1_012(partial_tuple: RelationDictionary, nodes: list) -> "tuple[RelationDictionary, float]":

    t1 = time.perf_counter()
    try:
        fitch_tree = algorithm_one(relations=partial_tuple, nodes=nodes, order=(0, 1, 2))
        t2 = time.perf_counter()
    except (NoSatRelation, NotFitchSatError):
        t2 = time.perf_counter()
        # error case
        return None, t2 - t1

    result_relations = cotree_to_rel(fitch_tree)

    return result_relations, t2 - t1


def pipeline_algo1_120(partial_tuple: RelationDictionary, nodes: list) -> "tuple[RelationDictionary, float]":

    t1 = time.perf_counter()
    try:
        fitch_tree = algorithm_one(relations=partial_tuple, nodes=nodes, order=(1, 2, 0))
        t2 = time.perf_counter()
    except (NoSatRelation, NotFitchSatError):
        t2 = time.perf_counter()
        # error case
        return None, t2 - t1

    result_relations = cotree_to_rel(fitch_tree)

    return result_relations, t2 - t1


def pipeline_algo1_210(partial_tuple: RelationDictionary, nodes: list) -> "tuple[RelationDictionary, float]":

    t1 = time.perf_counter()
    try:
        fitch_tree = algorithm_one(relations=partial_tuple, nodes=nodes, order=(2, 1, 0))
        t2 = time.perf_counter()
    except (NoSatRelation, NotFitchSatError):
        t2 = time.perf_counter()
        # error case
        return None, t2 - t1

    result_relations = cotree_to_rel(fitch_tree)

    return result_relations, t2 - t1


def pipeline_algo2(partial_tuple: RelationDictionary, nodes: list) -> "tuple[RelationDictionary, float]":

    weights: WeightedRelationDictionary = convert_to_weighted_relations(relations=partial_tuple, nodes=nodes)

    t1 = time.perf_counter()
    try:
        result_relations = algorithm_two(
            V=nodes,
            variables_empty=weights[0],
            variables_bi=weights[1],
            variables_uni=weights['d']
        )
        t2 = time.perf_counter()
    except (NoSatRelation, NotFitchSatError):
        t2 = time.perf_counter()
        # error case
        return None, t2 - t1

    return result_relations, t2 - t1
