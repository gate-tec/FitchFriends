import pandas as pd
import json

from fitch_graph_praktikum.nicolas.graph_io import load_dataframe, save_dataframe


def get_metrics_only(filename: str, excel_format: bool = False):
    base_frame = load_dataframe(filename=filename)

    col_names = [
        x for x in base_frame.columns if str(x).endswith('Sym_Diff') or str(x).endswith('Duration_(sek)')
    ]

    filtered_frame = pd.concat([
        pd.DataFrame(
            base_frame.ID.apply(lambda x: json.loads(x)).tolist(),
            index=base_frame.index, columns=['Nodes', 'Option', 'Instance']
        ).loc[:, ['Nodes', 'Option']],
        base_frame.loc[:, col_names].map(
            lambda x: str(x).replace('.', ',')
        ) if excel_format else base_frame.loc[:, col_names]
    ], axis=1)

    return filtered_frame, ['Nodes', 'Option']


if __name__ == "__main__":
    files = [
        'bm2_0.7_0.3_results.tsv',
        'bm2_0.7_0.4_results.tsv',
        'bm2_0.7_0.5_results.tsv'
    ]

    for file in files:
        filtered_frame, _ = get_metrics_only(file, True)
        save_dataframe(filename=f"excel_{file}", frame=filtered_frame)
