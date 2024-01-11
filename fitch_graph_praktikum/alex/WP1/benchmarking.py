import time
from fitch_graph_praktikum.nicolas.benchmark_WP1.benchmarking import pipeline_algo1_012, pipeline_algo1_120, \
    pipeline_algo1_210, pipeline_algo2
from fitch_graph_praktikum.nicolas.functions_partial_tuple import convert_to_relation_dict
from fitch_graph_praktikum.nicolas.graph_io import load_relations, save_dataframe
from fitch_graph_praktikum.util.lib import sym_diff
import pandas as pd
from fitch_graph_praktikum.alex.WP1.partial_cumulative_samples import create_partial_tuples_cumulativeLoss
from fitch_graph_praktikum.util.typing import RelationDictionary
from tqdm import tqdm
import json

import concurrent.futures


# auf alle originalen Fitch-Graphen 2. und 2b generieren mit je 10-90 Loss (nicht kumulativ)

def benchmark_algorithms(sampleID, loss, relations: "RelationDictionary", nodelist: list,
                         reference_relations: "RelationDictionary"):
    n = len(nodelist)

    relations_algo1_012, duration_algo1_012 = pipeline_algo1_012(relations, nodelist)
    if relations_algo1_012 is not None:
        sym_diff_algo1_012 = sym_diff(reference_relations, relations_algo1_012, n)
    else:
        sym_diff_algo1_012 = None
    results_012 = [sym_diff_algo1_012, duration_algo1_012, relations_algo1_012]

    relations_algo1_120, duration_algo1_120 = pipeline_algo1_120(relations, nodelist)
    if relations_algo1_120 is not None:
        sym_diff_algo1_120 = sym_diff(reference_relations, relations_algo1_120, n)
    else:
        sym_diff_algo1_120 = None
    results_120 = [sym_diff_algo1_120, duration_algo1_120, relations_algo1_120]

    relations_algo1_210, duration_algo1_210 = pipeline_algo1_210(relations, nodelist)
    if relations_algo1_210 is not None:
        sym_diff_algo1_210 = sym_diff(reference_relations, relations_algo1_210, n)
    else:
        sym_diff_algo1_210 = None
    results_210 = [sym_diff_algo1_210, duration_algo1_210, relations_algo1_210]

    relations_algo2, duration_algo2 = pipeline_algo2(relations, nodelist)
    if relations_algo2 is not None:
        sym_diff_algo2 = sym_diff(reference_relations, relations_algo2, n)
    else:
        sym_diff_algo2 = None
    results_2 = [sym_diff_algo2, duration_algo2, relations_algo2]

    results = {'ID': tuple(x for x in sampleID), 'Loss': loss}
    for algo_name, algo_results in zip(['Algo1_012', 'Algo1_120', 'Algo1_210', 'Algo2'],
                                       [results_012, results_120, results_210, results_2]):
        results[f"{algo_name}_Sym_Diff"] = round(algo_results[0],3)
        results[f"{algo_name}_Duration_(sek)"] = round(algo_results[1], 5)
        results[f"{algo_name}_Results_Rel_0"] = json.dumps(algo_results[2][0])
        results[f"{algo_name}_Results_Rel_1"] = json.dumps(algo_results[2][1])
        results[f"{algo_name}_Results_Rel_d"] = json.dumps(algo_results[2]['d'])

    return results


def benchmark_algorithms_on_all_samples(samples_DF: pd.DataFrame, reference=None):
    """
    Requires fixed format Dataframes as provided through alex/WP1/partial_cumulative_samples.
    It iterates over the whole DF and benchmarks for every relations_dictionary included.
    By giving the function a reference it only benchmarks the lines in the Dataframe, referring to the reference.
    The reference should be provided within a list, for example, as followed: [15,6,99] when using original graphs,
    else [15,99].

    Return: Dataframe of listet results from single benchmark results (as provided by benchmark_algorithms)
     """

    benchark_results = []
    if reference is not None:

        leaves = reference[0]

        default = 0
        x_option = default
        x_instance = default
        if len(reference) == 3:
            if 'x_options' not in samples_DF.columns:
                return print(
                    "If the reference states an x_option for the original Fitch-Graphs provided, a proper dataframe with x_options must be provided")
            else:
                x_option = reference[1]
                x_instance = reference[2]

        else:
            if 'x_options' in samples_DF.columns:
                return print(
                    "The Dataframe states x_options for the original Fitch-Graphs provided. The reference provided must contain one as well.")
            else:
                x_instance = reference[1]
                x_option = None

        if x_option is None:
            target_row = samples_DF.loc[
                samples_DF['Nodes'] == leaves & samples_DF['Sample'] == x_instance & samples_DF['Nominal_Loss'] == 0]
        else:
            target_row = samples_DF.loc[
            samples_DF['Nodes'] == leaves & samples_DF['x_options'] == x_option & samples_DF[
                'Sample'] == x_instance & samples_DF['Nominal_Loss'] == 0]

        reference_relations_string_0 = samples_DF.at[target_row, 'Relations_0']
        reference_relations_string_1 = samples_DF.at[target_row, 'Relations_1']
        reference_relations_string_d = samples_DF.at[target_row, 'Relations_d']

        reference_relation = convert_to_relation_dict(
            reference_relations_string_0,
            reference_relations_string_1,
            reference_relations_string_d
        )

        i = 1
        while x_instance == samples_DF.at[target_row + i, 'Sample']:

            nodelist = [x for x in range(leaves)]

            loss = samples_DF.at[target_row + i, 'Nominal_loss']

            loss_relations_string_0 = samples_DF.at[target_row + i, 'Relations_0']
            loss_relations_string_1 = samples_DF.at[target_row + i, 'Relations_1']
            loss_relations_string_d = samples_DF.at[target_row + i, 'Relations_d']

            loss_relations = convert_to_relation_dict(
                    loss_relations_string_0,
                    loss_relations_string_1,
                    loss_relations_string_d
                )
            
            sample_benchmark = benchmark_algorithms(reference, loss, loss_relations, nodelist, reference_relation)
            benchark_results.append(sample_benchmark)

            i = i + 1
        benchmark_df = pd.DataFrame(benchark_results)
        return benchmark_df

    else:  # if no reference is given: benchmark through all samples within the dataframe
        rows_total = len(samples_DF)
        benchmark_calls = []
        for i in range(rows_total):
            loss = samples_DF.at[i, 'Nominal_Loss']

            # new sample, since new samples in list are expected to start with loss = 0 to serve as referenc
            if loss == 0:
                reference_relations_string_0 = samples_DF.at[i, 'Relations_0']
                reference_relations_string_1 = samples_DF.at[i, 'Relations_1']
                reference_relations_string_d = samples_DF.at[i, 'Relations_d']

                reference_relations = convert_to_relation_dict(
                    reference_relations_string_0,
                    reference_relations_string_1,
                    reference_relations_string_d
                )

                nodes_count = samples_DF.at[i, 'Nodes']

                node_list = [x for x in range(nodes_count)]
                instace = samples_DF.at[i, 'Sample']
                if 'x_options' in samples_DF.columns:

                    x_option = samples_DF.at[i, 'x_options']
                    id = [nodes_count, x_option, instace]
                else:
                    id = [nodes_count, instace]
                continue
            else:
                relations_at_loss_string_0 = samples_DF.at[i, 'Relations_0']
                relations_at_loss_string_1 = samples_DF.at[i, 'Relations_1']
                relations_at_loss_string_d = samples_DF.at[i, 'Relations_d']

                relations_at_loss = convert_to_relation_dict(
                    relations_at_loss_string_0,
                    relations_at_loss_string_1,
                    relations_at_loss_string_d
                )
            try:
                benchmark_calls.append((id, loss, relations_at_loss, node_list, reference_relations))
                # instance_benchmark = benchmark_algorithms(id, loss, relations_at_loss, node_list, reference_relations)
                # benchark_results.append(instance_benchmark)
            except(UnboundLocalError):
                print("No reference relations at 0% loss have been provided. Input-Dataframe needs to be corrected ")
                return

        # speed up with concurrent futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=7) as executor:
            with tqdm(total=len(benchmark_calls), desc='Calls', disable=False) as progress:
                work_load = {executor.submit(
                    benchmark_algorithms,
                    benchmark_calls[i][0],
                    benchmark_calls[i][1],
                    benchmark_calls[i][2],
                    benchmark_calls[i][3],
                    benchmark_calls[i][4]
                ): i for i in range(len(benchmark_calls))}

                for future in work_load:
                    future.add_done_callback(lambda p: progress.update())
                for future in concurrent.futures.as_completed(work_load):
                    index = work_load[future]
                    try:
                        instance_benchmark = future.result()
                    except Exception as exc:
                        # catch thrown exceptions and store for printing later on
                        progress.write('%d generated an exception: %s' % (index, exc))
                    else:
                        benchark_results.append(instance_benchmark)

    benchmark_df = pd.DataFrame.from_records(benchark_results)
    benchmark_df = benchmark_df.sort_values(by=['ID', 'Loss']).reset_index(drop=True)
    benchmark_df.loc[:, 'ID'] = benchmark_df.loc[:, 'ID'].apply(lambda x: list(y for y in x))
    return benchmark_df


if __name__ == '__main__':
    test_initial_relation = load_relations(15, 5, 9)
    sample = create_partial_tuples_cumulativeLoss(15, 5, 9, test_initial_relation, 0.1)
    test_df = pd.DataFrame(sample)

    benchmark_df = benchmark_algorithms_on_all_samples(test_df)
    save_dataframe('benchmark_results_cumulative_sample.tsv', benchmark_df)
