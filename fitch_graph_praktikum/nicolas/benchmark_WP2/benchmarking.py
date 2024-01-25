import copy
import json
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

from fitch_graph_praktikum.alex.WP2.full_weighted_relations import generate_full_weighted_relations
from fitch_graph_praktikum.util.lib import partition_heuristic_scaffold, sym_diff
from fitch_graph_praktikum.util.typing import WeightedRelationDictionary, RelationDictionary
from fitch_graph_praktikum.nicolas.graph_io import load_relations, delete_file, save_dataframe

from fitch_graph_praktikum.nicolas.framework_WP2.framework_functions import (
    bi_partition_random,
    bi_partition_greedy_avg, bi_partition_greedy_sum,
    bi_partition_louvain_sum_edge_cut, bi_partition_louvain_average_edge_cut,
    bi_partition_louvain_average_edge_cut_mod,
    bi_partition_leiden_average_edge_cut_gamma1_theta1, bi_partition_leiden_sum_edge_cut_gamma1_theta1,
    bi_partition_leiden_average_edge_cut_gamma1_7_theta001, bi_partition_leiden_sum_edge_cut_gamma1_7_theta001,
    bi_partition_leiden_average_edge_cut_mod_gamma1_theta1, bi_partition_leiden_average_edge_cut_mod_gamma1_7_theta001
)
from fitch_graph_praktikum.alex.WP2.weight_scoring_for_partitioning import average_weight_scoring, sum_weight_scoring


def _serialize_weights(weights: dict[tuple[int, int], float]) -> "str":
    return json.dumps([{'k': k, 'v': v} for k, v in weights.items()])


def _deserialize_weights(weight_str: str) -> "dict[tuple[int, int], float]":
    return {tuple(entry['k']): entry['v'] for entry in json.loads(weight_str)}


def benchmark_algos_on_graph(
        sampleID, number_of_nodes: int, relations: RelationDictionary, mu_TP, mu_FP,
        input_full_weight_relations: list[str] = None
) -> "tuple[Any, Any]":

    # generate random weights
    full_weight_relations: WeightedRelationDictionary = generate_full_weighted_relations(
        number_of_nodes=number_of_nodes, relations=relations,
        distribution_TP=np.random.normal, parameters_TP=[mu_TP, 0.1],
        distribution_FP=np.random.normal, parameters_FP=[mu_FP, 0.1]
    ) if input_full_weight_relations is None else {
        0: _deserialize_weights(input_full_weight_relations[0]),
        1: _deserialize_weights(input_full_weight_relations[1]),
        'd': _deserialize_weights(input_full_weight_relations[2])
    }

    # record base
    weight_record = {
        'ID': sampleID,
        'Weights_0': _serialize_weights(full_weight_relations[0]),
        'Weights_1': _serialize_weights(full_weight_relations[1]),
        'Weights_d': _serialize_weights(full_weight_relations['d'])
    }
    record = {
        'ID': sampleID
    }

    functions = [
        ('RandomAvg', bi_partition_random, average_weight_scoring, False, False),
        ('GreedyAvg', bi_partition_greedy_avg, average_weight_scoring, False, False),
        ('GreedySum', bi_partition_greedy_sum, average_weight_scoring, False, False),
        ('LouvainAvg', bi_partition_louvain_average_edge_cut, average_weight_scoring, False, False),
        ('LouvainAvgMod', bi_partition_louvain_average_edge_cut_mod, average_weight_scoring, False, False),
        # ('LouvainSum', bi_partition_louvain_sum_edge_cut, sum_weight_scoring, False, False),
        ('LeidenAvg_1_1', bi_partition_leiden_average_edge_cut_gamma1_theta1, average_weight_scoring, False, False),
        ('LeidenAvgMod_1_1', bi_partition_leiden_average_edge_cut_mod_gamma1_theta1, average_weight_scoring, False, False),
        # ('LeidenSum_1_1', bi_partition_leiden_sum_edge_cut_gamma1_theta1, sum_weight_scoring, False, False),
        ('LeidenAvg_1_7_001', bi_partition_leiden_average_edge_cut_gamma1_7_theta001, average_weight_scoring, False, False),
        ('LeidenAvgMod_1_7_001', bi_partition_leiden_average_edge_cut_mod_gamma1_7_theta001, average_weight_scoring, False, False),
        # ('LeidenSum_1_7_001', bi_partition_leiden_sum_edge_cut_gamma1_7_theta001, sum_weight_scoring, False, False),
        ('GreedyAvg_med', bi_partition_greedy_avg, average_weight_scoring, False, False),
        ('GreedySum_med', bi_partition_greedy_sum, average_weight_scoring, False, False),
        ('LouvainAvg_med', bi_partition_louvain_average_edge_cut, average_weight_scoring, True, True),
        ('LouvainAvgMod_med', bi_partition_louvain_average_edge_cut_mod, average_weight_scoring, True, True),
        # ('LouvainSum_med', bi_partition_louvain_sum_edge_cut, sum_weight_scoring, True, True),
        ('LeidenAvg_1_1_med', bi_partition_leiden_average_edge_cut_gamma1_theta1, average_weight_scoring, True, True),
        ('LeidenAvgMod_1_1', bi_partition_leiden_average_edge_cut_mod_gamma1_theta1, average_weight_scoring, True, True),
        # ('LeidenSum_1_1_med', bi_partition_leiden_sum_edge_cut_gamma1_theta1, sum_weight_scoring, True, True),
        ('LeidenAvg_1_7_001_med', bi_partition_leiden_average_edge_cut_gamma1_7_theta001, average_weight_scoring, True, True),
        ('LeidenAvgMod_1_7_001', bi_partition_leiden_average_edge_cut_mod_gamma1_7_theta001, average_weight_scoring, True, True),
        # ('LeidenSum_1_7_001_med', bi_partition_leiden_sum_edge_cut_gamma1_7_theta001, sum_weight_scoring, True, True),
    ]

    for name, part_func, score_func, median, reciprocal in functions:
        input_data = copy.deepcopy({0: [], 1: [], "d": []})
        t1 = time.perf_counter()
        predicted_relations: RelationDictionary = partition_heuristic_scaffold(
            uni_weighted=full_weight_relations['d'],
            bi_weighted=full_weight_relations[1],
            empty_weighted=full_weight_relations[0],
            relations=input_data,
            nodes=[x for x in range(number_of_nodes)],
            partition_function=part_func,
            scoring_function=score_func,
        )
        t2 = time.perf_counter()

        if len(set(predicted_relations[0]).intersection(predicted_relations[1])) > 0 or \
                len(set(predicted_relations[0]).intersection(predicted_relations['d'])) > 0 or \
                len(set(predicted_relations[1]).intersection(predicted_relations['d'])) > 0:
            raise ValueError(f'Critical 1 - {sampleID}: {name}')

        difference = sym_diff(relations, predicted_relations, n=number_of_nodes)

        record.update(**{
            f"{name}_Sym_Diff": round(difference, 3),
            f"{name}_Duration_(sek)": round(t2 - t1, 5),
            f"{name}_Results_Rel_0": json.dumps(copy.deepcopy(list(sorted(set(predicted_relations[0]))))),
            f"{name}_Results_Rel_1": json.dumps(copy.deepcopy(list(sorted(set(predicted_relations[1]))))),
            f"{name}_Results_Rel_d": json.dumps(copy.deepcopy(list(sorted(set(predicted_relations['d'])))))
        })

    return weight_record, record


def benchmark_all_stored_graphs(mu_TP: float, mu_FP: float, safety_steps: int = 500):
    nodes = [10, 15, 20, 25, 30]
    x_options = list(range(7))
    x_instances = list(range(100))

    instances = [(num_nodes, option, i) for num_nodes in nodes for option in x_options for i in x_instances]

    weight_results = []
    benchmark_results = []

    threads = max(os.cpu_count() - 1, 2)
    print(f'Multiprocessing on {threads} cores:')
    call_counter = 0
    last_stable_frame = None
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        with tqdm(total=len(instances), desc='Calls', disable=False) as progress:
            work_load = {executor.submit(
                benchmark_algos_on_graph,
                instance,
                instance[0],
                load_relations(instance[0], instance[1], instance[2]),
                mu_TP,
                mu_FP,
                None
            ): i for i, instance in enumerate(instances)}

            for future in work_load:
                future.add_done_callback(lambda p: progress.update())
            for future in concurrent.futures.as_completed(work_load):
                index = work_load[future]
                try:
                    weight_result, benchmark_result = future.result()
                    call_counter += 1
                except Exception as exc:
                    # catch thrown exceptions and store for printing later on
                    progress.write('%d generated an exception: %s' % (index, exc))
                else:
                    weight_results.append(weight_result)
                    benchmark_results.append(benchmark_result)
                    if call_counter % safety_steps == 0:
                        # safety store data
                        next_frame = f"temp-{time.perf_counter()}-{call_counter}"
                        save_dataframe(f"{next_frame}_rec", pd.DataFrame.from_records(benchmark_results))
                        save_dataframe(f"{next_frame}_weight", pd.DataFrame.from_records(weight_results))
                        if last_stable_frame is not None:
                            delete_file(f"{last_stable_frame}_rec")
                            delete_file(f"{last_stable_frame}_weight")
                        last_stable_frame = next_frame

    benchmark_df = pd.DataFrame.from_records(benchmark_results)
    benchmark_df = benchmark_df.sort_values(by=['ID']).reset_index(drop=True)
    benchmark_df.loc[:, 'ID'] = benchmark_df.loc[:, 'ID'].apply(lambda x: list(y for y in x))

    weight_df = pd.DataFrame.from_records(weight_results)
    weight_df = weight_df.sort_values(by=['ID']).reset_index(drop=True)
    weight_df.loc[:, 'ID'] = weight_df.loc[:, 'ID'].apply(lambda x: list(y for y in x))

    if last_stable_frame is not None:
        delete_file(f"{last_stable_frame}_rec")
        delete_file(f"{last_stable_frame}_weight")

    return benchmark_df, weight_df
