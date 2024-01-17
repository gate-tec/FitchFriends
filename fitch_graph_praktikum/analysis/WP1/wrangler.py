import pandas as pd
import json

from fitch_graph_praktikum.nicolas.graph_io import load_dataframe


def get_metrics_only(filename: str):
    base_frame = load_dataframe(filename=filename)

    col_names = [
        x for x in base_frame.columns if str(x).endswith('Sym_Diff') or str(x).endswith('Duration_(sek)')
    ]

    if len(json.loads(base_frame.loc[0, 'ID'])) == 3:
        # original data frame
        id_columns = ['Nodes', 'Option', 'Instance']
        sub_id_columns = ['Nodes', 'Option']
        group_columns = ['Nodes', 'Option', 'Loss']
    else:
        # random data frame
        id_columns = ['Nodes', 'Instance']
        sub_id_columns = ['Nodes']
        group_columns = ['Nodes', 'Loss']

    filtered_frame = pd.concat([
        pd.DataFrame(
            base_frame.ID.apply(lambda x: json.loads(x)).tolist(),
            index=base_frame.index, columns=id_columns
        ).loc[:, sub_id_columns],
        base_frame.loc[:, ['Loss'] + col_names]
    ], axis=1)

    return filtered_frame, group_columns


if __name__ == "__main__":
    filtered_frame, group_columns = get_metrics_only('bm_original_fitch_samples_nicolas.tsv')
    # filtered_frame, group_columns = get_metrics_only('bm_original_fitch_samples_no_direct_nicolas.tsv')
    # filtered_frame, group_columns = get_metrics_only('bm_random_fitch_samples_nicolas.tsv')
    # filtered_frame, group_columns = get_metrics_only('bm_random_fitch_samples_no_direct_nicolas.tsv')

    mean_frame = filtered_frame\
        .groupby(by=group_columns)\
        .filter(lambda x: len(x) > 50)\
        .groupby(by=group_columns)\
        .mean()

    print(mean_frame.to_string())
