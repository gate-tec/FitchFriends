import os.path
from os.path import abspath, dirname
from inspect import getsourcefile

from typing import Dict, Union, List, Set
import pickle as pk


_options = [
    "D0.3_L0.3_H0.9", "D0.5_L0.5_H0.5", "D0.5_L0.5_H0.25", "D0.5_L0.5_H1.0",
    "D0.5_L1.0_H0.25", "D0.25_L0.5_H1.0", "D1.0_L0.5_H0.25"
]
_instances = list(range(100))
_rel_path = os.path.join(dirname(abspath(getsourcefile(lambda: 0))), '../../graph-prak-GFH')

def load_relations(x_leaves, x_options, x_instances) -> Dict[Union[int, str], List]:
    path = f"{_rel_path}/n{x_leaves}/{_options[x_options]}/{_options[x_options]}_n{x_leaves}_{_instances[x_instances]}"
    with open(f"{path}/biRelations.pkl", 'rb') as bi_file:
        bi_relations = pk.load(bi_file)
    with open(f"{path}/uniRelations.pkl", 'rb') as uni_file:
        uni_relations = pk.load(uni_file)
    with open(f"{path}/emptyRelations.pkl", 'rb') as empty_file:
        empty_relations = pk.load(empty_file)
    return {0: empty_relations, 1: bi_relations, "d": uni_relations}
