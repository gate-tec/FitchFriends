import os.path
from os.path import abspath, dirname
from inspect import getsourcefile

from typing import TYPE_CHECKING, Literal, Union, overload
from fitch_graph_praktikum.util.typing import RelationDictionary
import pickle as pk
import pandas as pd

if TYPE_CHECKING:
    from typing import Any, Dict, Optional


__all__ = ["load_relations", "save_dataframe", "load_dataframe"]


_options = [
    "D0.3_L0.3_H0.9", "D0.5_L0.5_H0.5", "D0.5_L0.5_H0.25", "D0.5_L0.5_H1.0",
    "D0.5_L1.0_H0.25", "D0.25_L0.5_H1.0", "D1.0_L0.5_H0.25"
]
_instances = list(range(100))
_rel_path = os.path.join(dirname(abspath(getsourcefile(lambda: 0))), '../../graph-prak-GFH')


def load_relations(
        x_leaves: Literal[10, 15, 20, 25, 30],
        x_options: int,
        x_instances: int
) -> "RelationDictionary":
    """
    Load relations `E_0`, `E_1`, and `E_d` from specified xenologous graph.

    Parameters
    ----------
    x_leaves: {10, 15, 20, 25, 30}
        Number of leaves in the xenologous graph.
    x_options: int
        Specific configuration (must be between 0 and 6, inclusive).
    x_instances: int
        Specific instance of xenologous graph (must be between 0 and 99, inclusive).

    Returns
    -------
    RelationDictionary
        Dictionary of relations with keys `0`, `1`, and `'d'` respectively.

    Raises
    ------
    ValueError
        if `x_options` or `x_instances` is outside the given boundaries.
    """
    if not (0 <= x_options <= 6):
        raise ValueError(f'x_options must be between 0 and 6, inclusive. Got {x_options} instead')
    if not (0 <= x_instances <= 99):
        raise ValueError(f'x_instances must be between 0 and 99, inclusive. Got {x_instances} instead')

    path = f"{_rel_path}/n{x_leaves}/{_options[x_options]}/{_options[x_options]}_n{x_leaves}_{_instances[x_instances]}"
    with open(f"{path}/biRelations.pkl", 'rb') as bi_file:
        bi_relations = pk.load(bi_file)
    with open(f"{path}/uniRelations.pkl", 'rb') as uni_file:
        uni_relations = pk.load(uni_file)
    with open(f"{path}/emptyRelations.pkl", 'rb') as empty_file:
        empty_relations = pk.load(empty_file)
    return {0: empty_relations, 1: bi_relations, "d": uni_relations}


def save_dataframe(filename: str, frame: pd.DataFrame, absolute: bool = False):
    path = abspath(filename)
    if not absolute:
        path = os.path.join(_rel_path, filename)
    if not os.path.exists(dirname(path)):
        os.makedirs(dirname(path))

    frame.to_csv(path, sep='\t', header=True, index=True)


def load_dataframe(filename: str, absolute: bool = False):
    path = abspath(filename)
    if not absolute:
        path = os.path.join(_rel_path, filename)
    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path, sep='\t', header=0, index_col=0)
    except pd.errors.ParserError:
        return None
