import time
from fitch_graph_praktikum.nicolas.benchmark_WP1.benchmarking import pipeline_algo1_012, pipeline_algo1_120, \
    pipeline_algo1_210, pipeline_algo2
from fitch_graph_praktikum.nicolas.graph_io import load_relations
from fitch_graph_praktikum.util.lib import sym_diff
import pandas as pd
from partial_cumulative_samples import create_partial_tuples_cumulativeLoss
from fitch_graph_praktikum.util.typing import RelationDictionary


# auf alle originalen Fitch-Graphen 2. und 2b generieren mit je 10-90 Loss (nicht kumulativ)

def benchmark_algorithms(sampleID, loss, relations: RelationDictionary, nodelist: list,
                         reference_relations: RelationDictionary):
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

    results = {'ID': sampleID, 'Loss': loss, 'Algo1_012': results_012, 'Algo1_120': results_120,
               'Algo1_210': results_210,
               'Algo2': results_2}

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
        x_option = reference[1]
        try:
            x_instance = reference[2]
        except(IndexError):
            x_instance = reference[1]
            x_option = None

        if x_option is None:
            target_row = samples_DF.loc[
                samples_DF['Nodes'] == leaves & samples_DF['Sample'] == x_instance & samples_DF['Nominal_Loss'] == 0]
        else:
            target_row = samples_DF.loc[
                samples_DF['Nodes'] == leaves & samples_DF['x_options'] == x_option & samples_DF[
                    'Sample'] == x_instance & samples_DF['Nominal_Loss'] == 0]

        reference_relation = samples_DF.at[target_row, 'Relations']
        i = 1
        while x_instance == samples_DF.at[target_row + i, 'Sample']:

            if x_option is not None:
                nodelist = [str(x) for x in range(leaves)]
            else:
                nodelist = [x for x in range(leaves)]

            loss = samples_DF.at[target_row + i, 'Nominal_loss']
            loss_relations = samples_DF.at[[target_row + i, 'Relations']]
            sample_benchmark = benchmark_algorithms(reference, loss, loss_relations, nodelist, reference_relation)
            benchark_results.append(sample_benchmark)

            i = i + 1
        benchark_df = pd.DataFrame(benchark_results)
        return benchark_df

    else:  # if no reference is given: benchmark through all samples within the dataframe
        rows_total = len(samples_DF)
        for i in range(rows_total):
            loss = samples_DF.at[i, 'Nominal_Loss']

            # new sample, since new samples in list are expected to start with loss = 0 to serve as referenc
            if loss == 0:
                reference_relations = samples_DF.at[i, 'Relations']

                nodes_count = samples_DF.at[i, 'Nodes']


                instace = samples_DF.at[i, 'Sample']
                if 'x_options' in samples_DF.columns:
                    node_list = [str(x) for x in range(nodes_count)]
                    x_option = samples_DF.at[i, 'x_options']
                    id = [nodes_count, x_option, instace]
                else:
                    node_list = [x for x in range(nodes_count)]
                    id = [nodes_count, instace]
                continue
            else:
                relations_at_loss = samples_DF.at[i, 'Relations']
            try:
                instance_benchmark = benchmark_algorithms(id, loss, relations_at_loss, node_list, reference_relations)
                benchark_results.append(instance_benchmark)
            except(UnboundLocalError):
                print("No reference relations at 0% loss have been provided. Input-Dataframe needs to be corrected ")
                return
        benchark_df = pd.DataFrame(benchark_results)
        return benchark_df


if __name__ == '__main__':
    test_initial_relation = load_relations(15, 5, 9)
    sample = create_partial_tuples_cumulativeLoss(15, 5, 9, test_initial_relation, 0.1)
    test_df = pd.DataFrame(sample)

    print(benchmark_algorithms_on_all_samples(test_df))
