from create_partial_tuples import create_partial_tuples
from create_random_dicograph import create_random_dicograph
from fitch_graph_praktikum.util.helper_functions import NoSatRelation, NotFitchSatError
from fitch_graph_praktikum.util.lib import graph_to_rel, algorithm_one
import pandas as pd

if __name__ == '__main__':
    node_sizes = [10, 15, 20, 25, 30]
    sample_size = 100
    samples = []
    losses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    partial_samples = []
    samples_df = pd.DataFrame(
        columns=['Nodes', 'Sample', 'Nominal_Loss', 'Effective_Loss', 'Is_Fitch_Sat', 'Relations'])

    for nodes in node_sizes:
        nodelist = list(range(nodes))
        for i in range(0, sample_size):
            dicograph = create_random_dicograph(nodes)
            relations_0 = graph_to_rel(dicograph)
            try:
                is_fitchtree = algorithm_one(relations_0, nodelist, order=(0, 1, 2))
                fitchsat = True
            except (NoSatRelation, NotFitchSatError):
                fitchsat = False

            samples_df = samples_df._append(
                {'Nodes': nodes, 'Sample': i, 'Nominal_Loss': 0, 'Effective_Loss': 0, 'Is_Fitch_Sat': fitchsat,
                 'Relations': relations_0},
                ignore_index=True)

            for loss in losses:
                relations_at_loss, eff_loss = create_partial_tuples(relations_0, loss)

                try:
                    is_fitchtree = algorithm_one(relations_at_loss, nodelist, order=(0, 1, 2))
                    fitchsat = True
                except (NoSatRelation, NotFitchSatError):
                    fitchsat = False

                samples_df = samples_df._append(
                    {'Nodes': nodes, 'Sample': i, 'Nominal_Loss': loss, 'Effective_Loss': eff_loss,
                     'Is_Fitch_Sat': fitchsat, 'Relations': relations_at_loss}, ignore_index=True)
        samples_df.to_csv("partialsamples.tsv", sep="\t")
