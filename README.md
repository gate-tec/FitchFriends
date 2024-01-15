# fitch-graph-prak - Team: FitchFriends

Practical course researching reconstruction of xenologous relations and heuristics for the weighted Fitch graph problem.

## Development Environment

The supplemental data is required to be located within
[graph-prak-GFH](graph-prak-GFH/).


## Function Record

### General

- `load_dataframe` in [fitch_graph_praktikum.nicolas.graph_io](fitch_graph_praktikum/nicolas/graph_io.py):
  load a `.tsv` file as a pandas DataFrame
- `save_dataframe` in [fitch_graph_praktikum.nicolas.graph_io](fitch_graph_praktikum/nicolas/graph_io.py):
  save a pandas DataFrame to a `.tsv` file
- `load_relations` in [fitch_graph_praktikum.nicolas.graph_io](fitch_graph_praktikum/nicolas/graph_io.py):
  load relations from stored data

### WP1

- `get_reduced_relations` in [fitch_graph_praktikum.nicolas.functions_partial_tuple](fitch_graph_praktikum/nicolas/functions_partial_tuple.py):
  get all (possible) relations with specified information loss (with or without clearing `E_d`)
- `get_random_cograph_relations` in [fitch_graph_praktikum.nicolas.generator](fitch_graph_praktikum/nicolas/generator.py):
  get all (possible) relations with specified information loss for a random (Fitch) graph
- pipeline functions in [fitch_graph_praktikum.nicolas.benchmark_WP1.benchmarking](fitch_graph_praktikum/nicolas/benchmark_WP1/benchmarking.py):
  singular pipeline functions for each algorithm (variant)
- `benchmark_algorithms_on_all_samples` in [fitch_graph_praktikum.alex.WP1.benchmarking](fitch_graph_praktikum/alex/WP1/benchmarking.py):
  run the benchmarking for a given DataFrame

### WP2

- `generate_full_weighted_relations` in [fitch_graph_praktikum.alex.WP2.full_weighted_relations](fitch_graph_praktikum/alex/WP2/full_weighted_relations.py):
  generate full sets of weights for a relation dictionary