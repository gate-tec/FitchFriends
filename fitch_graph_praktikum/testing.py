import random
import time

from fitch_graph_praktikum.util.lib import generate_weights, algorithm_two
from fitch_graph_praktikum.nicolas.functions_partial_tuple import convert_to_weighted_relations


# Init some nodes
nodes = [0, 1, 2]

# Init some partial relations
relation = {
    0: [],
    1: [(0, 1), (1, 0)],
    "d": [(1, 2)]
}

weights = convert_to_weighted_relations(relations=relation, nodes=nodes)

for k, v in weights.items():
    print(f"{k}: {v}")

t1 = time.perf_counter()
run2 = algorithm_two(
    V=nodes,
    variables_empty=weights[0],
    variables_bi=weights[1],
    variables_uni=weights['d']
)
run_time = time.perf_counter() - t1
print(f"Elapsed time: {run_time:.2f}")
for k, v in run2.items():
    print(f"{k}: {v}")
