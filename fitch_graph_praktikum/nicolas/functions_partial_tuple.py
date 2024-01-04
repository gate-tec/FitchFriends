import math
from typing import TYPE_CHECKING, Literal, Union, overload
from fitch_graph_praktikum.util.typing import RelationDictionary
import random

if TYPE_CHECKING:
    from typing import Any, Dict, Optional


__all__ = ["delete_information", "get_reduced_relations"]


def delete_information(
        relations: RelationDictionary,
        percent_delete: int = 10
) -> "tuple[RelationDictionary, float]":
    e_0 = set(relations[0])
    e_1 = set(relations[1])
    e_d = set(relations["d"])

    total_relations = e_0.union(e_1).union(e_d)
    num_total = len(total_relations)
    num_remove = math.ceil(num_total * (percent_delete / 100.0))

    while num_remove > 0:
        relation = random.sample(total_relations, 1)[0]
        total_relations.remove(relation)
        num_remove -= 1
        if relation in e_0 or relation in e_1:
            total_relations.remove((relation[1], relation[0]))
            num_remove -= 1

    # TODO: Optimize
    return {
        0: list(sorted(e_0.intersection(total_relations))),
        1: list(sorted(e_1.intersection(total_relations))),
        "d": list(sorted(e_d.intersection(total_relations)))
    }, 1 - len(total_relations) / num_total


def get_reduced_relations(
        relations: RelationDictionary,
        list_percent_delete: list[int]
) -> "Dict[int, RelationDictionary]":
    relation_dict = {}

    for percent_delete in list_percent_delete:
        reduced_relations, _ = delete_information(relations=relations, percent_delete=percent_delete)
        relation_dict[percent_delete] = reduced_relations

    return relation_dict
