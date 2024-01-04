from fitch_graph_praktikum.nicolas.functions_partial_tuple import get_reduced_relations
from fitch_graph_praktikum.nicolas.graph_io import load_relations
from fitch_graph_praktikum.nicolas.tree_functions import create_tree, construct_random_cograph
from fitch_graph_praktikum.util.lib import cotree_to_rel, rel_to_fitch, check_fitch_graph, graph_to_rel

import networkx as nx

from typing import TYPE_CHECKING, Literal, Union, overload
from fitch_graph_praktikum.util.typing import RelationDictionary

if TYPE_CHECKING:
    from typing import Any, Dict, Optional


__all__ = ["get_stored_cograph_relations", "get_random_cograph_relations"]


def get_stored_cograph_relations(
        num_nodes: Literal[10, 15, 20, 25, 30],
        x_options: int,
        x_instances: int
) -> "tuple[nx.DiGraph, RelationDictionary, Dict[int, RelationDictionary], bool]":
    """
    Load relations `E_0`, `E_1`, and `E_d` from specified xenologous graph.

    Parameters
    ----------
    num_nodes: {10, 15, 20, 25, 30}
        Number of nodes in the xenologous cograph.
    x_options: int
        Specific configuration (must be between 0 and 6, inclusive).
    x_instances: int
        Specific instance of xenologous graph (must be between 0 and 99, inclusive).

    Returns
    -------
    tuple[nx.DiGraph, RelationDictionary, Dict[int, RelationDictionary], bool]
        Tuple containing:

        - the stored cograph
        - the corresponding relations
        - a dictionary mapping each information loss (10, 20, ..., 90) to the respective relations
        - whether the cograph is Fitch.

    Raises
    ------
    ValueError
        if `x_options` or `x_instances` is outside the given boundaries.
    """
    nodes = [str(x) for x in range(num_nodes)]
    rels = load_relations(x_leaves=num_nodes, x_options=x_options, x_instances=x_instances)

    cograph = rel_to_fitch(rels, nodes)

    reduced_rels = get_reduced_relations(rels, list(range(10, 99, 10)))

    return cograph, rels, reduced_rels, check_fitch_graph(cograph)


def get_random_cograph_relations(
        num_nodes: Literal[10, 15, 20, 25, 30],
        top_down: bool = False
) -> "tuple[nx.DiGraph, RelationDictionary, Dict[int, RelationDictionary], bool]":
    """
    Construct random cograph including its reduced sets of relations.

    Parameters
    ----------
    num_nodes: {10, 15, 20, 25, 30}
        Number of nodes in the cograph.
    top_down: bool
        Whether the corresponding cotree should be constructed top-down (default: False).

    Returns
    -------
    tuple[nx.DiGraph, RelationDictionary, Dict[int, RelationDictionary], bool]
        Tuple containing:

        - the resulting cograph
        - the corresponding relations
        - a dictionary mapping each information loss (10, 20, ..., 90) to the respective relations
        - whether the cograph is Fitch.
    """
    nodes = list(range(num_nodes))

    if top_down:
        tree = create_tree(nodes=nodes)
        rels: RelationDictionary = cotree_to_rel(tree)

        cograph = rel_to_fitch(rels, nodes)
    else:
        cograph = construct_random_cograph(nodes=nodes)
        rels: RelationDictionary = graph_to_rel(cograph)

    reduced_rels = get_reduced_relations(rels, list(range(10, 99, 10)))

    return cograph, rels, reduced_rels, check_fitch_graph(cograph)
