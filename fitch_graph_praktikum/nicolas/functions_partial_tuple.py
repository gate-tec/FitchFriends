import math
from typing import Dict, Union, List, Set, Tuple
import random


def delete_information(relations: Dict[Union[int, str], List], percent_delete: int = 10) -> Tuple[Dict[Union[int, str], List], float]:
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
    }, len(total_relations) / num_total

