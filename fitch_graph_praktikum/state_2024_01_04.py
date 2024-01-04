# Authors:
#   - Nicolas Handke    https://github.com/gate-tec

from fitch_graph_praktikum.nicolas.generator import get_random_cograph_relations, get_stored_cograph_relations

import networkx as nx


# Load "D0.25_L0.5_H1.0_n10_0"
fitch_graph, full_rels, reduced_rels, is_fitch_sat = get_stored_cograph_relations(
    num_nodes=10,
    x_options=5,
    x_instances=0
)

print(is_fitch_sat)
for k, v in reduced_rels.items():
    print(f"{k}: {v}")


# Create random cograph (Method 2), ~95% Fitch-SAT
cograph, full_rels, reduced_rels, is_fitch_sat = get_random_cograph_relations(
    num_nodes=10,
    top_down=True
)

print(is_fitch_sat)
for k, v in reduced_rels.items():
    print(f"{k}: {v}")
