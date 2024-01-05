import math
from typing import TYPE_CHECKING, Literal, Union, overload
from fitch_graph_praktikum.util.typing import RelationDictionary, WeightedRelationDictionary
import random

if TYPE_CHECKING:
    from typing import Any, Dict, Optional


__all__ = ["delete_information", "get_reduced_relations", "convert_to_weighted_relations"]


@overload
def delete_information(
        relations: RelationDictionary,
        percent_delete: int = 10,
        clear_e_direct: Literal[False] = False
) -> "tuple[RelationDictionary, float]":
    ...


@overload
def delete_information(
        relations: RelationDictionary,
        percent_delete: int = 10,
        clear_e_direct: Literal[True] = False
) -> "tuple[RelationDictionary, float, int]":
    ...


def delete_information(
        relations: RelationDictionary,
        percent_delete: int = 10,
        clear_e_direct: bool = False
) -> "Union[tuple[RelationDictionary, float], tuple[RelationDictionary, float, int]]":
    e_0_half = [(x, y) for (x, y) in relations[0] if x < y]
    e_1_half = [(x, y) for (x, y) in relations[1] if x < y]
    e_d = relations["d"]

    total_relations = e_0_half + e_1_half + e_d
    num_total = len(total_relations)
    num_keep = num_total - math.ceil(num_total * (percent_delete / 100.0))

    effective_loss = round((1 - num_keep / num_total) * 100, 1)

    tuples_to_keep = random.sample(total_relations, num_keep)
    rels: RelationDictionary = {0: [], 1: [], 'd': []}

    for rel in tuples_to_keep:
        if rel in e_0_half:
            rels[0].append(rel)
            rels[0].append((rel[1], rel[0]))
        elif rel in e_1_half:
            rels[1].append(rel)
            rels[1].append((rel[1], rel[0]))
        else:
            rels['d'].append(rel)

    return rels, effective_loss


def get_reduced_relations(
        relations: RelationDictionary,
        list_percent_delete: list[int]
) -> "Dict[int, tuple[RelationDictionary, float]]":
    relation_dict = {}

    for percent_delete in list_percent_delete:
        reduced_relations, loss = delete_information(relations=relations, percent_delete=percent_delete)
        relation_dict[percent_delete] = (reduced_relations, loss)

    return relation_dict


def convert_to_weighted_relations(
        relations: RelationDictionary,
        nodes: list
) -> "WeightedRelationDictionary":
    weights: WeightedRelationDictionary = {0: {}, 1: {}, 'd': {}}

    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            node1, node2 = nodes[i], nodes[j]
            if (node1, node2) in relations[0]:
                weights[0][(node1, node2)] = 100
                weights[0][(node2, node1)] = 100
                weights[1][(node1, node2)] = -100
                weights[1][(node2, node1)] = -100
                weights['d'][(node1, node2)] = -100
                weights['d'][(node2, node1)] = -100
            elif (node1, node2) in relations[1]:
                weights[0][(node1, node2)] = -100
                weights[0][(node2, node1)] = -100
                weights[1][(node1, node2)] = 100
                weights[1][(node2, node1)] = 100
                weights['d'][(node1, node2)] = -100
                weights['d'][(node2, node1)] = -100
            elif (node1, node2) in relations['d']:
                weights[0][(node1, node2)] = -100
                weights[0][(node2, node1)] = -100
                weights[1][(node1, node2)] = -100
                weights[1][(node2, node1)] = -100
                weights['d'][(node1, node2)] = 100
                weights['d'][(node2, node1)] = -100
            elif (node2, node1) in relations['d']:
                weights[0][(node1, node2)] = -100
                weights[0][(node2, node1)] = -100
                weights[1][(node1, node2)] = -100
                weights[1][(node2, node1)] = -100
                weights['d'][(node1, node2)] = -100
                weights['d'][(node2, node1)] = 100
            else:
                weights[0][(node1, node2)] = -100
                weights[0][(node2, node1)] = -100
                weights[1][(node1, node2)] = -100
                weights[1][(node2, node1)] = -100
                weights['d'][(node1, node2)] = -100
                weights['d'][(node2, node1)] = -100

    return weights
