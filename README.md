# fitch-graph-prak

#### As most of the functions are described in detail in the practical script, here we focus on some basic usage of the functions provided. You can also find the code provided here in the main of lib.py.

The following snip shows the structure of relations that we used throughout our framework. We model a tuple of relations by a dictionary. The corresponding relationship types are accessible by different keys:
* 1 for bidirectional edges
* 0 for empty edges
* d for unidirectional edges

We represent weighted relations by a dictionary, where the keys are the corresponding edges. For example, ```uni_weighted[(0, 1)]``` contains the weight for having a unidirectional edge between node 0 and node 1. 

```
    # Init some nodes
    nodes = [0, 1, 2]

    # Init some partial relations
    relation = {
        0: [],
        1: [(0, 1), (1, 0)],
        "d": [(1, 2)]
    }

    # Init weights for unidirectional relations, bidirectional relations, and empty relations
    uni_weighted = {
        (1, 2): 100,
        (0, 1): -100,
        (1, 0): -100,
        (2, 1): -100,
        (0, 2): -100,
        (2, 0): -100
    }

    bi_weighted = {
        (0, 1): 100,
        (1, 0): 100,
        (1, 2): -100,
        (2, 1): -100,
        (0, 2): -100,
        (2, 0): -100
    }

    empty_weighted = {
        (0, 1): -100,
        (1, 0): -100,
        (1, 2): -100,
        (2, 1): -100,
        (0, 2): -100,
        (2, 0): -100
    }
```


The following snip shows how to call the ```algorithm_one``` function to complete the partial set we just defined above. We also need to specify an order of how Rules S1, S2, and S3 are applied. The last argument, e.g. ```(1, 2, 0)``` specifies in which order the rules are applied, and in this case, we would first check S2, then S3, and last S1. Take note that ```algorithm_one``` returns a cotree.

```
    # Compute a Cotree using the partial set defined prior.
    fitch_cotree_210 = algorithm_one(relation, nodes, (1, 2, 0))

    # Compute a Cotree using the partial set defined prior but with a different order of Rules.
    fitch_cotree_012 = algorithm_one(relation, nodes, (0, 1, 2))
```

The cotrees ```algorithm_one``` returns can easily be parsed into a relations dictionary by using ```cotree_to_rel```. If you want to use the function to translate your generated cotrees into relations/graphs, we give more details about their requirements below.

```
    # Parse the cotrees just computed into a relations dictionary.
    fitch_relations_210 = cotree_to_rel(fitch_cotree_210)
    fitch_relations_012 = cotree_to_rel(fitch_cotree_012)
```

The function ```algorithm_two``` takes three weighted relations dictionaries as input and tries to find a maximum weight solution for the weighted fitch optimization problem. It returns a dictionary of relations that can also be translated into a graph object by calling ```rel_to_fitch```

```
    # Run the greedy algorithm on the weighted relations initialized above.
    fitch_relations_greedy = algorithm_two(nodes, uni_weighted, bi_weighted, empty_weighted)
    fitch_graph_greedy = rel_to_fitch(fitch_relations_greedy, nodes)
```

We provided a basic framework to generate weighted relations using different distributions/random generators. It takes a list of tuples (relations), a number generator (like ```random.uniform``` or ```random.random```), and parameters for the random generator as input and returns a dictionary of weighted relations. The keyword argument ```symmetric``` is used to specify whether two symmetric edges, e.g. (0, 1) and (1, 0) should have the same or different weights. When generating weights for WP2, you should use something similar to this function to generate edge-weights/weighted relations depending on whether a relation is present in a given Fitch graph or not.

```
    # Generate weights for "1" with the random.uniform sampling between 1.0 and 1.5
    test_weights_bi = generate_weights(relation[1], random.uniform, [1.0, 1.5], symmetric=True)

    # Generate weights for "d" with the random.random generator. As it takes no arguments, the parameters are empty.
    test_weights_uni = generate_weights(relation["d"], random.random, [], symmetric=False)
```

Given an arbitrary directed graph (here the Fitch graph reconstructed from a partial set by ```algorithm_one```), the function ```check_fitch_graph``` can be used to verify whether it is a Fitch graph. It uses the forbidden subgraph characterization to compute whether a graph is a Fitch graph in O(n^3).

```
    # Check if the graph reconstructed from fitch_relations_210 is fitch-sat
    fitch_graph_210 = rel_to_fitch(fitch_relations_210, nodes)
    is_fitch = check_fitch_graph(fitch_graph_210)
```

A quick node to the function ```cotree_to_rel```can also be used to utilize cotrees for the random cograph generation of WP1. You need a special structure in case you want to use this to parse randomly generated cotrees. Each node in the cotree needs to have a label "symbol". Inner Nodes need the "symbol" "u", "b", or "e" where u encodes ->1 nodes, b encodes 1 nodes, and e encodes 0 nodes. Leaves are expected to have an integer value of 0...n, where n is the total number of nodes in the underlying graph. For ->1 nodes, the order of children is realized by the node ids, i.e. the integer value used to initiate a node. The code below creates a cotree which encodes the graph G=({0, 1, 2}, {(0, 1), (0, 2), (1, 2)}).

```
    # Initialize an empty cotree.
    cotree = nx.DiGraph()

    # Add one inner node ->1 with three leaves 0, 1, and 2.
    cotree.add_node(0, symbol="u")
    cotree.add_node(1, symbol=0)
    cotree.add_node(2, symbol=1)
    cotree.add_node(3, symbol=2)

    cotree.add_edge(0, 1)
    cotree.add_edge(0, 2)
    cotree.add_edge(0, 3)

    # Translate cotree to relations
    decoded_cotree = cotree_to_rel(cotree)
```

The partition heuristic framework is implemented in the ```partition_heuristic_scaffold``` function. Besides weighted relation dictionaries, it also requires a partition function and a scoring function as input. The scoring function (here simp_score) is expected to take an input of a partition and a dictionary of weighted relations, e.g., for the case of simp_score - ```simp_score([0], [1, 2], uni_weighted)```. The partition function should only take a networkx.DiGraph as input.

```
    # That is how you can call the partition heuristic, simp_part and simp_score still need to be implemented.
    # fitch_relations_partition = partition_heuristic_scaffold(uni_weighted, bi_weighted, empty_weighted, nodes, simp_part, simp_score)
```
