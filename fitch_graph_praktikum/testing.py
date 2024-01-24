import json
import random
import time

import pandas as pd

from fitch_graph_praktikum.nicolas.benchmark_WP2.benchmarking import benchmark_algos_on_graph
from fitch_graph_praktikum.util.lib import generate_weights, algorithm_two, algorithm_one, cotree_to_rel
from fitch_graph_praktikum.nicolas.functions_partial_tuple import convert_to_weighted_relations
from fitch_graph_praktikum.nicolas.graph_io import load_relations, load_dataframe


# Init some nodes
frame: pd.DataFrame = load_dataframe("bm2_0.7_0.3_results.tsv")
col_names = [
        x for x in frame.columns if str(x).endswith('Sym_Diff')
    ]

columns = ['ID', 'LeidenSum_1_7_001_Sym_Diff', 'LeidenSum_1_7_001_Results_Rel_0',
           'LeidenSum_1_7_001_Results_Rel_1', 'LeidenSum_1_7_001_Results_Rel_d']

filtered: pd.DataFrame = frame.loc[frame['LeidenSum_1_7_001_Sym_Diff'] > 1.0, columns].reset_index(drop=True)

# print(filtered.head(2).to_string())

for i, row in filtered.iterrows():
    if i >= 1:
        break

    id_ref = tuple(x for x in json.loads(row['ID']))
    reference_relation = load_relations(x_leaves=id_ref[0], x_options=id_ref[1], x_instances=id_ref[2])
    #
    # rel_0 = list(sorted(set(tuple(x for x in elem) for elem in json.loads(row['LeidenSum_1_7_001_Results_Rel_0']))))
    # rel_1 = list(sorted(set(tuple(x for x in elem) for elem in json.loads(row['LeidenSum_1_7_001_Results_Rel_1']))))
    # rel_d = list(sorted(set(tuple(x for x in elem) for elem in json.loads(row['LeidenSum_1_7_001_Results_Rel_d']))))
    #
    # print(reference_relation[0])
    # print(rel_0)
    # print()
    # print(reference_relation[1])
    # print(rel_1)
    # print()
    # print(reference_relation['d'])
    # print(rel_d)
    # print()

    result = benchmark_algos_on_graph(
        sampleID=id_ref, number_of_nodes=id_ref[0], relations=reference_relation, mu_TP=0.7, mu_FP=0.3
    )
    print(result[len(result) - 1])
