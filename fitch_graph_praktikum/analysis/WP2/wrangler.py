import pandas as pd
import json

from fitch_graph_praktikum.nicolas.graph_io import load_dataframe


def get_metrics_only(filename: str):
    base_frame = load_dataframe(filename=filename)

    col_names = [
        x for x in base_frame.columns if str(x).endswith('Sym_Diff') or str(x).endswith('Duration_(sek)')
    ]

    filtered_frame = pd.concat([
        pd.DataFrame(
            base_frame.ID.apply(lambda x: json.loads(x)).tolist(),
            index=base_frame.index, columns=['Nodes', 'Option', 'Instance']
        ).loc[:, ['Nodes', 'Option']],
        base_frame.loc[:, col_names]
    ], axis=1)

    return filtered_frame, ['Nodes', 'Option']


if __name__ == "__main__":
    # filtered_frame, group_columns = get_metrics_only('bm2_0.7_0.3_results.tsv')
    # filtered_frame, group_columns = get_metrics_only('bm2_0.7_0.4_results.tsv')
    filtered_frame, group_columns = get_metrics_only('bm2_0.7_0.5_results.tsv')

    mean_frame = filtered_frame.groupby(by=group_columns).mean()

    print(mean_frame.to_string())
