import math
from typing import TYPE_CHECKING, Literal, Union, overload
from fitch_graph_praktikum.util.typing import RelationDictionary, WeightedRelationDictionary
from fitch_graph_praktikum.util.lib import algorithm_two
import random

if TYPE_CHECKING:
    from typing import Any, Dict, Optional


__all__ = ["delete_information", "get_reduced_relations", "convert_to_weighted_relations"]


class NotEnoughDataError(Exception):
    """Raised when deletion of E_d would overstep the specified information loss."""
    pass


def delete_information(
        relations: RelationDictionary,
        percent_delete: int = 10,
        clear_e_direct: bool = False
) -> "tuple[RelationDictionary, float]":
    e_0_half = [(x, y) for (x, y) in relations[0] if x < y]
    e_1_half = [(x, y) for (x, y) in relations[1] if x < y]
    e_d = relations["d"]

    total_relations = e_0_half + e_1_half + e_d
    num_total = len(total_relations)

    num_delete = math.ceil(num_total * (percent_delete / 100.0))
    num_keep = num_total - num_delete

    if clear_e_direct:
        if len(e_d) > num_delete:
            raise NotEnoughDataError
        num_delete -= len(e_d)
        total_relations = e_0_half + e_1_half

    effective_loss = round((1 - num_keep / num_total) * 100, 1)

    if num_delete > 0:

        tuples_to_keep = random.sample(total_relations, num_keep)
        rels: RelationDictionary = {0: [], 1: [], 'd': []}

        for rel in tuples_to_keep:
            if rel in e_0_half:
                rels[0].append(rel)
                rels[0].append((rel[1], rel[0]))
            elif rel in e_1_half:
                rels[1].append(rel)
                rels[1].append((rel[1], rel[0]))
            elif not clear_e_direct:
                # sanity check
                rels['d'].append(rel)
            else:
                raise ValueError('Should not happen')
    else:
        rels: RelationDictionary = {0: relations[0], 1: relations[1], 'd': []}

    return rels, effective_loss


def get_reduced_relations(
        relations: RelationDictionary,
        list_percent_delete: list[int],
        clear_e_direct: bool = False
) -> "Dict[int, tuple[RelationDictionary, float]]":
    relation_dict = {}

    for percent_delete in list_percent_delete:
        try:
            reduced_relations, loss = delete_information(
                relations=relations,
                percent_delete=percent_delete,
                clear_e_direct=clear_e_direct
            )
            relation_dict[percent_delete] = (reduced_relations, loss)
        except NotEnoughDataError:
            pass

    return relation_dict


def convert_to_weighted_relations(
        relations: RelationDictionary,
        nodes: list
) -> "WeightedRelationDictionary":
    """
    Converts a given partial relation into a weighted relation for `algorithm_two`.

    Parameters
    ----------
    relations: RelationDictionary
        partial tuple of relations.
    nodes: list
        List of all nodes.

    Returns
    -------
    WeightedRelationDictionary
        completed tuple of weighted relations.

    See Also
    --------
    algorithm_two
    """
    weights: WeightedRelationDictionary = {0: {}, 1: {}, 'd': {}}
    
    weight_contained = 100  # m_0
    weight_not_contained = -100  # -m_0
    weight_unknown = 0

    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            node1, node2 = nodes[i], nodes[j]
            if (node1, node2) in relations[0]:
                weights[0][(node1, node2)] = weight_contained
                weights[0][(node2, node1)] = weight_contained
                weights[1][(node1, node2)] = weight_not_contained
                weights[1][(node2, node1)] = weight_not_contained
                weights['d'][(node1, node2)] = weight_not_contained
                weights['d'][(node2, node1)] = weight_not_contained
            elif (node1, node2) in relations[1]:
                weights[0][(node1, node2)] = weight_not_contained
                weights[0][(node2, node1)] = weight_not_contained
                weights[1][(node1, node2)] = weight_contained
                weights[1][(node2, node1)] = weight_contained
                weights['d'][(node1, node2)] = weight_not_contained
                weights['d'][(node2, node1)] = weight_not_contained
            elif (node1, node2) in relations['d']:
                weights[0][(node1, node2)] = weight_not_contained
                weights[0][(node2, node1)] = weight_not_contained
                weights[1][(node1, node2)] = weight_not_contained
                weights[1][(node2, node1)] = weight_not_contained
                weights['d'][(node1, node2)] = weight_contained
                weights['d'][(node2, node1)] = weight_not_contained
            elif (node2, node1) in relations['d']:
                weights[0][(node1, node2)] = weight_not_contained
                weights[0][(node2, node1)] = weight_not_contained
                weights[1][(node1, node2)] = weight_not_contained
                weights[1][(node2, node1)] = weight_not_contained
                weights['d'][(node1, node2)] = weight_not_contained
                weights['d'][(node2, node1)] = weight_contained
            else:
                weights[0][(node1, node2)] = weight_unknown
                weights[0][(node2, node1)] = weight_unknown
                weights[1][(node1, node2)] = weight_unknown
                weights[1][(node2, node1)] = weight_unknown
                weights['d'][(node1, node2)] = weight_unknown
                weights['d'][(node2, node1)] = weight_unknown

    return weights
