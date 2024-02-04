import pandas as pd
import numpy as np
import json

from fitch_graph_praktikum.nicolas.graph_io import load_dataframe, save_dataframe
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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


color_set = [x for x in mcolors.TABLEAU_COLORS.values()]


def boxplot(df, columns, labels, title, path):
    plt.figure()

    colors = color_set[:len(columns)]

    ticks = list(range(10, 35, 5))

    for i in range(len(columns)):
        data = [
            list(df.loc[df['Nodes'] == num_nodes, columns[i]]) for num_nodes in ticks
        ]
        deviation = (i - ((len(columns) - 1) / 2.0)) * 0.8
        bpi = plt.boxplot(
            data,
            positions=np.array(range(len(data))) * len(columns) + deviation,
            sym='',
            widths=0.6
        )

        plt.setp(bpi['boxes'], color=colors[i])
        plt.setp(bpi['whiskers'], color=colors[i])
        plt.setp(bpi['caps'], color=colors[i])
        plt.setp(bpi['medians'], color=colors[i])

        plt.plot([], c=colors[i], label=labels[i])

    plt.legend()
    plt.title(title)

    plt.xticks(range(0, len(ticks) * len(columns), len(columns)), [str(x) for x in ticks])
    plt.xlim(-len(columns), len(ticks) * len(columns))
    plt.xlabel('# Nodes')
    plt.ylim(0, 1)
    plt.ylabel('Sym Diff')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(path)


def boxplot_overall(df1, df2, df3, columns, labels, title, path):
    plt.figure()

    colors = color_set[:len(columns)]

    ticks = ["(0.7, 0.3)", "(0.7, 0.4)", "(0.7, 0.5)"]

    for i in range(len(columns)):
        data = [
            list(df1.loc[:, columns[i]]),
            list(df2.loc[:, columns[i]]),
            list(df3.loc[:, columns[i]]),
        ]
        deviation = (i - ((len(columns) - 1) / 2.0)) * 0.8
        bpi = plt.boxplot(
            data,
            positions=np.array(range(len(data))) * len(columns) + deviation,
            sym='',
            widths=0.6
        )

        plt.setp(bpi['boxes'], color=colors[i])
        plt.setp(bpi['whiskers'], color=colors[i])
        plt.setp(bpi['caps'], color=colors[i])
        plt.setp(bpi['medians'], color=colors[i])

        plt.plot([], c=colors[i], label=labels[i])

    plt.legend()
    plt.title(title)

    plt.xticks(range(0, len(ticks) * len(columns), len(columns)), [str(x) for x in ticks])
    plt.xlim(-len(columns), len(ticks) * len(columns))
    plt.xlabel(b'\xCE\xBC for TP & FP'.decode('utf-8'))
    plt.ylim(0, 1)
    plt.ylabel('Sym Diff')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(path)


def barplot_time(df, columns, labels, title, path):
    plt.figure()
    colors = color_set[:len(columns)]

    ticks = list(range(10, 35, 5))

    for i in range(len(columns)):
        data = [
            list(df.loc[df['Nodes'] == num_nodes, columns[i]]) for num_nodes in ticks
        ]
        deviation = (i - ((len(columns) - 1) / 2.0)) * 0.8
        bpi = plt.boxplot(
            data,
            positions=np.array(range(len(data))) * len(columns) + deviation,
            sym='',
            widths=0.6
        )

        plt.setp(bpi['boxes'], color=colors[i])
        plt.setp(bpi['whiskers'], color=colors[i])
        plt.setp(bpi['caps'], color=colors[i])
        plt.setp(bpi['medians'], color=colors[i])

        plt.plot([], c=colors[i], label=labels[i])

    plt.legend()
    plt.title(title)

    plt.xticks(range(0, len(ticks) * len(columns), len(columns)), [str(x) for x in ticks])
    plt.xlim(-len(columns), len(ticks) * len(columns))
    plt.xlabel('# Nodes')
    # plt.ylim(bottom=0.0)
    plt.yscale('log')
    plt.ylabel('run time (sec)')
    plt.tight_layout()
    plt.savefig(path)


def load_excel_frame(path: str) -> "pd.DataFrame":
    frame = load_dataframe(path)
    diff_names = [
        x for x in frame.columns if str(x).endswith('Sym_Diff')
    ]
    frame.loc[:, diff_names] = frame.loc[:, diff_names].map(
        lambda x: float(str(x).replace(',', '.'))
    )
    time_names = [
        x for x in frame.columns if str(x).endswith('Duration_(sek)')
    ]
    frame.loc[:, time_names] = frame.loc[:, time_names].map(
        lambda x: float(str(x).replace(',', '.'))
    )
    return frame


if __name__ == "__main__":
    options = [
        "D0.5_L0.5_H0.25", "D0.5_L1.0_H0.25", "D1.0_L0.5_H0.25", "D0.5_L0.5_H0.5",
        "D0.3_L0.3_H0.9", "D0.25_L0.5_H1.0", "D0.5_L0.5_H1.0"
    ]

    frame_data = [
        ('0.7_0.3', '(0.7, 0.3)', 'excel_bm2_0.7_0.3_results.tsv'),
        ('0.7_0.4', '(0.7, 0.4)', 'excel_bm2_0.7_0.4_results.tsv'),
        ('0.7_0.5', '(0.7, 0.5)', 'excel_bm2_0.7_0.5_results.tsv')
    ]
    frames = tuple(load_excel_frame(frame_data[x][2]) for x in range(len(frame_data)))
    frame0, frame1, frame2 = frames

    ##########################
    # Select combination #####
    ##########################

    # algos = ['GreedyAvg', 'GreedySum', 'LouvainAvg', 'LeidenAvg_1_1', 'LeidenAvg_1_7_001']
    # labels = ['Greedy (avg)', 'Greedy (sum)', 'Louvain (avg)', 'Leiden (1, 1) (avg)', b'Leiden (\xE2\x85\x90, 0.01) (avg)'.decode('utf-8')]
    # group = "Avg_v_BM"

    algos = ['LouvainAvg', 'LouvainAvg_med', 'LeidenAvg_1_1', 'LeidenAvg_1_1_med', 'LeidenAvg_1_7_001', 'LeidenAvg_1_7_001_med']
    labels = ['Louvain', 'Louvain med', 'Leiden (1, 1)', 'Leiden med (1, 1)', b'Leiden (\xE2\x85\x90, 0.01)'.decode('utf-8'), b'Leiden med (\xE2\x85\x90, 0.01)'.decode('utf-8')]
    group = "Avg6_v_BM"

    ####################
    # Create Plots #####
    ####################

    boxplot_overall(
        df1=frame0, df2=frame1, df3=frame2,
        columns=[f"{x}_Sym_Diff" for x in algos],
        labels=labels,
        title=f'WP2',
        path=f"bm2_{group}_all.png"
    )
    for selected_option in range(len(options)):
        boxplot_overall(
            df1=frame0.loc[frame0['Option'] == selected_option, :],
            df2=frame1.loc[frame1['Option'] == selected_option, :],
            df3=frame2.loc[frame2['Option'] == selected_option, :],
            columns=[f"{x}_Sym_Diff" for x in algos],
            labels=labels,
            title=f'{options[selected_option]} on WP2',
            path=f"bm2_{group}_{options[selected_option]}.png"
        )

    for selected_frame in range(len(frames)):
        frame = frames[selected_frame]
        boxplot(
            df=frame,
            columns=[f"{x}_Sym_Diff" for x in algos],
            labels=labels,
            title=f'{frame_data[selected_frame][1]} with Scoring=Avg',
            path=f"bm2_{group}_all_{frame_data[selected_frame][0]}.png"
        )
        for selected_option in range(len(options)):
            boxplot(
                df=frame.loc[frame['Option'] == selected_option, :],
                columns=[f"{x}_Sym_Diff" for x in algos],
                labels=labels,
                title=f'{options[selected_option]} on {frame_data[selected_frame][1]} with Scoring=Avg',
                path=f"bm2_{group}_{options[selected_option]}_{frame_data[selected_frame][0]}.png"
            )
