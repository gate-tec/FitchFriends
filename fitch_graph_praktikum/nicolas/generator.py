from fitch_graph_praktikum.nicolas.functions_partial_tuple import get_reduced_relations
from fitch_graph_praktikum.nicolas.graph_io import load_relations
from fitch_graph_praktikum.nicolas.tree_functions import create_tree, construct_random_cograph
from fitch_graph_praktikum.util.helper_functions import NoSatRelation, NotFitchSatError
from fitch_graph_praktikum.util.lib import cotree_to_rel, rel_to_fitch, graph_to_rel, algorithm_one, check_fitch_graph

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
) -> "tuple[nx.DiGraph, Dict[int, tuple[RelationDictionary, float, bool]]]":
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
        - a dictionary mapping each information loss (0, 10, 20, ..., 90) to the respective relations,
          their effective loss, and whether these relations are Fitch-SAT.

    Raises
    ------
    ValueError
        if `x_options` or `x_instances` is outside the given boundaries.
    """
    nodes = [str(x) for x in range(num_nodes)]
    rels = load_relations(x_leaves=num_nodes, x_options=x_options, x_instances=x_instances)

    cograph = rel_to_fitch(rels, nodes)

    reduced_rels = get_reduced_relations(rels, list(range(10, 99, 10)))
    reduced_rels[0] = (rels, 0.0)
    reduced_rels = dict(sorted(reduced_rels.items()))

    return cograph, {k: (rels, loss, True) for k, (rels, loss) in reduced_rels.items()}


@Exception
class AlgorithmOneError:
    """Raised when sanity check of algorithm 1 fails."""
    pass


def get_random_cograph_relations(
        num_nodes: Literal[10, 15, 20, 25, 30],
        top_down: bool = True,
        do_sanity_check: bool = False,
        ensure_base_fitch: bool = True
) -> "tuple[nx.DiGraph, Dict[int, tuple[RelationDictionary, float, bool]]]":
    """
    Construct random cograph including its reduced sets of relations.

    Parameters
    ----------
    num_nodes: {10, 15, 20, 25, 30}
        Number of nodes in the cograph.
    top_down: bool
        Whether the corresponding cotree should be constructed top-down (default: False).
    do_sanity_check: bool
        Whether the result of the fitch-SAT decision should be double-checked (default: False).
    ensure_base_fitch: bool
        Guarantee that the generated base cograph is Fitch (default: True).
        Automatically sets top_down to True.

    Returns
    -------
    tuple[nx.DiGraph, RelationDictionary, Dict[int, RelationDictionary], bool]
        Tuple containing:

        - the resulting cograph
        - a dictionary mapping each information loss (0, 10, 20, ..., 90) to the respective relations,
          their effective loss, and whether these relations are Fitch-SAT.

    Raises
    ------
    AlgorithmOneError
        if sanity check of Algorithm 1 fails (only if `do_sanity_check` is True).
    """
    nodes = list(range(num_nodes))
    if ensure_base_fitch:
        # ensure_base_fitch only works for top_down == True
        top_down = True

    if top_down:
        while True:
            tree = create_tree(nodes=nodes)
            rels: RelationDictionary = cotree_to_rel(tree)

            cograph = rel_to_fitch(rels, nodes)

            # If ensure_base_fitch == True: loop until graph is Fitch
            if not ensure_base_fitch or check_fitch_graph(cograph):
                break
    else:
        cograph = construct_random_cograph(nodes=nodes)
        rels: RelationDictionary = graph_to_rel(cograph)

    reduced_rels = get_reduced_relations(rels, list(range(10, 99, 10)))
    reduced_rels[0] = (rels, 0.0)
    reduced_rels = dict(sorted(reduced_rels.items()))

    mapped_reduced_rels = {}
    for k, (rels, loss) in reduced_rels.items():
        if ensure_base_fitch:
            # if base graph is Fitch, then all relations are Fitch-SAT
            fitch_sat = True
        else:
            try:
                fitch_tree = algorithm_one(rels, nodes, order=(0, 1, 2))

                if do_sanity_check:
                    # sanity check if resulting cotree actually explains a fitch graph
                    fitch_graph = rel_to_fitch(cotree_to_rel(fitch_tree), nodes)
                    if not check_fitch_graph(fitch_graph):
                        raise AlgorithmOneError
                fitch_sat = True
            except (NoSatRelation, NotFitchSatError):
                fitch_sat = False

        mapped_reduced_rels[k] = (rels, loss, fitch_sat)

    return cograph, mapped_reduced_rels
