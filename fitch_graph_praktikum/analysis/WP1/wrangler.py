from typing import Literal

import pandas as pd
import numpy as np
import json

from fitch_graph_praktikum.nicolas.graph_io import load_dataframe, save_dataframe
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


color_set = [x for x in mcolors.TABLEAU_COLORS.values()]


def get_metrics_only(filename: str, excel_format: bool = False):
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
        base_frame.loc[:, ['Loss'] + col_names].map(
            lambda x: str(x).replace('.', ',')
        ) if excel_format else base_frame.loc[:, ['Loss'] + col_names]
    ], axis=1)

    return filtered_frame, group_columns


def create_loss_to_nodes_plots(df, key: str = "Algo1_012"):
    pivot_012_sym_nodes = pd.pivot_table(
        df,
        values=f'{key}_Sym_Diff', index=['Nodes'], columns=['Loss'], aggfunc='mean'
    )

    plt.title(f"Mean Sym Diff of {key} towards Loss")
    for i, row in pivot_012_sym_nodes.iterrows():
        plt.plot(list(range(10, 100, 10)), np.asarray(row), label=f"{i} Nodes")
    plt.ylim(bottom=0.0, top=1.0)
    plt.xlabel("% Loss")
    plt.ylabel("Sym Diff")
    plt.legend()
    plt.show()


def plot_box_loss_to_nodes(df, key: str, title: str, path: str):
    plt.figure()
    node_list = list(range(10, 35, 5))

    colors = color_set[:len(node_list)]

    ticks = list(range(10, 100, 10))

    for i in range(len(node_list)):
        data = [
            list(df.loc[(df['Nodes'] == node_list[i]) & (df['Loss'] == loss), f"{key}_Sym_Diff"]) for loss in ticks
        ]
        deviation = (i - ((len(node_list) - 1) / 2.0)) * 0.8
        bpi = plt.boxplot(
            data,
            positions=np.array(range(len(data))) * len(node_list) + deviation,
            sym='',
            widths=0.6
        )

        plt.setp(bpi['boxes'], color=colors[i])
        plt.setp(bpi['whiskers'], color=colors[i])
        plt.setp(bpi['caps'], color=colors[i])
        plt.setp(bpi['medians'], color=colors[i])

        plt.plot([], c=colors[i], label=f"{node_list[i]} Nodes")

    plt.legend()
    plt.title(title)

    plt.xticks(range(0, len(ticks) * len(node_list), len(node_list)), [str(x) for x in ticks])
    plt.xlim(-len(node_list), len(ticks) * len(node_list))
    plt.xlabel('Loss (%)')
    plt.ylim(bottom=0.0, top=1.0)
    plt.grid(axis='y')
    # plt.yscale('log')
    plt.ylabel('Sym Diff')
    plt.tight_layout()
    plt.savefig(path)


def plot_box_loss_to_nodes_rt(df, key: str, title: str, path: str):
    plt.figure()
    node_list = list(range(10, 35, 5))

    colors = color_set[:len(node_list)]

    ticks = list(range(10, 100, 10))

    for i in range(len(node_list)):
        data = [
            list(df.loc[(df['Nodes'] == node_list[i]) & (df['Loss'] == loss), f"{key}_Duration_(sek)"]) for loss in ticks
        ]
        deviation = (i - ((len(node_list) - 1) / 2.0)) * 0.8
        bpi = plt.boxplot(
            data,
            positions=np.array(range(len(data))) * len(node_list) + deviation,
            sym='',
            widths=0.6
        )

        plt.setp(bpi['boxes'], color=colors[i])
        plt.setp(bpi['whiskers'], color=colors[i])
        plt.setp(bpi['caps'], color=colors[i])
        plt.setp(bpi['medians'], color=colors[i])

        plt.plot([], c=colors[i], label=f"{node_list[i]} Nodes")

    plt.legend()
    plt.title(title)

    plt.xticks(range(0, len(ticks) * len(node_list), len(node_list)), [str(x) for x in ticks])
    plt.xlim(-len(node_list), len(ticks) * len(node_list))
    plt.xlabel('Loss (%)')
    # plt.ylim(bottom=0.0, top=1.0)
    plt.grid(axis='y')
    plt.yscale('log')
    plt.ylabel('run time (sec)')
    plt.tight_layout()
    plt.savefig(path)


def boxplot(df, columns, labels, title, path, y_axis: Literal['Sym Diff', 'run time (sec)'] = 'Sym Diff'):
    plt.figure()

    colors = color_set[:len(columns)]

    ticks = list(range(10, 100, 10))

    for i in range(len(columns)):
        data = [
            list(df.loc[df['Loss'] == loss, columns[i]]) for loss in ticks
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
    plt.xlabel('loss (%)')
    if y_axis == 'Sym Diff':
        plt.ylim(0, 1)
    else:
        plt.yscale('log')
    plt.ylabel(y_axis)
    plt.grid(axis='y')
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
        ('bm_orig', 'Original', 'excel_bm_original_fitch_samples_nicolas.tsv'),
        ('bm_orig_no_dir', "Original (no E" + b'\xE2\x82\x81\xE2\x83\x97'.decode('utf-8') + ")", 'excel_bm_original_fitch_samples_no_direct_nicolas.tsv'),
        ('bm_rand', 'Random', 'excel_bm_random_fitch_samples_nicolas.tsv'),
        ('bm_rand_no_dir', "Random (no E" + b'\xE2\x82\x81\xE2\x83\x97'.decode('utf-8') + ")", 'excel_bm_random_fitch_samples_no_direct_nicolas.tsv')
    ]
    files = [
        'excel_bm_original_fitch_samples_nicolas.tsv',
        'excel_bm_original_fitch_samples_no_direct_nicolas.tsv',
        'excel_bm_random_fitch_samples_nicolas.tsv',
        'excel_bm_random_fitch_samples_no_direct_nicolas.tsv'
    ]
    
    frames = tuple(load_excel_frame(frame_data[x][2]) for x in range(len(frame_data)))
    
    for selected_frame, frame in enumerate(frames):

        create_loss_to_nodes_plots(df=frame, key="Algo1_012")
        create_loss_to_nodes_plots(df=frame, key="Algo1_120")
        create_loss_to_nodes_plots(df=frame, key="Algo1_210")
        create_loss_to_nodes_plots(df=frame, key="Algo2")
    
        plot_box_loss_to_nodes(
            df=frame,
            key="Algo1_012",
            title=f"{frame_data[selected_frame][1]} Algo1 (0,1,2)",
            path=f"{frame_data[selected_frame][0]}_Algo1_012_sym_diff.png"
        )
        plot_box_loss_to_nodes(
            df=frame,
            key="Algo1_120",
            title=f"{frame_data[selected_frame][1]} Algo1 (1,2,0)",
            path=f"{frame_data[selected_frame][0]}_Algo1_120_sym_diff.png"
        )
        plot_box_loss_to_nodes(
            df=frame,
            key="Algo1_210",
            title=f"{frame_data[selected_frame][1]} Algo1 (2,1,0)",
            path=f"{frame_data[selected_frame][0]}_Algo1_210_sym_diff.png"
        )
        plot_box_loss_to_nodes(
            df=frame,
            key="Algo2",
            title=f"{frame_data[selected_frame][1]} Algo2",
            path=f"{frame_data[selected_frame][0]}_Algo2_sym_diff.png"
        )

        plot_box_loss_to_nodes_rt(
            df=frame,
            key="Algo1_012",
            title=f"{frame_data[selected_frame][1]} Algo1 (0,1,2)",
            path=f"{frame_data[selected_frame][0]}_Algo1_012_run_time.png"
        )
        plot_box_loss_to_nodes_rt(
            df=frame,
            key="Algo1_120",
            title=f"{frame_data[selected_frame][1]} Algo1 (1,2,0)",
            path=f"{frame_data[selected_frame][0]}_Algo1_120_run_time.png"
        )
        plot_box_loss_to_nodes_rt(
            df=frame,
            key="Algo1_210",
            title=f"{frame_data[selected_frame][1]} Algo1 (2,1,0)",
            path=f"{frame_data[selected_frame][0]}_Algo1_210_run_time.png"
        )
        plot_box_loss_to_nodes_rt(
            df=frame,
            key="Algo2",
            title=f"{frame_data[selected_frame][1]} Algo2",
            path=f"{frame_data[selected_frame][0]}_Algo2_run_time.png"
        )

        if selected_frame < 2:
            boxplot(
                df=frame,
                columns=["Algo1_012_Sym_Diff", "Algo1_120_Sym_Diff", "Algo1_210_Sym_Diff", "Algo2_Sym_Diff"],
                labels=["Algo1_012", "Algo1_120", "Algo1_210", "Algo2"],
                title=f'{frame_data[selected_frame][1]}  v. Loss',
                path=f"{frame_data[selected_frame][0]}_diff_loss.png"
            )
            for selected_option in range(len(options)):
                boxplot(
                    df=frame.loc[frame['Option'] == selected_option, :],
                    columns=["Algo1_012_Sym_Diff", "Algo1_120_Sym_Diff", "Algo1_210_Sym_Diff", "Algo2_Sym_Diff"],
                    labels=["Algo1_012", "Algo1_120", "Algo1_210", "Algo2"],
                    title=f'{frame_data[selected_frame][1]} {options[selected_option]} v. Loss',
                    path=f"{frame_data[selected_frame][0]}_{options[selected_option]}_diff_loss.png"
                )
        else:
            boxplot(
                df=frame,
                columns=["Algo1_012_Sym_Diff", "Algo1_120_Sym_Diff", "Algo1_210_Sym_Diff", "Algo2_Sym_Diff"],
                labels=["Algo1_012", "Algo1_120", "Algo1_210", "Algo2"],
                title=f'{frame_data[selected_frame][1]} v. Loss',
                path=f"{frame_data[selected_frame][0]}_diff_loss.png"
            )
    
        if selected_frame < 2:
            boxplot(
                df=frame,
                columns=["Algo1_012_Duration_(sek)", "Algo1_120_Duration_(sek)", "Algo1_210_Duration_(sek)", "Algo2_Duration_(sek)"],
                labels=["Algo1_012", "Algo1_120", "Algo1_210", "Algo2"],
                title=f'{frame_data[selected_frame][1]}  run time',
                path=f"{frame_data[selected_frame][0]}_run_time.png",
                y_axis='run time (sec)'
            )
            for selected_option in range(len(options)):
                boxplot(
                    df=frame.loc[frame['Option'] == selected_option, :],
                    columns=["Algo1_012_Duration_(sek)", "Algo1_120_Duration_(sek)", "Algo1_210_Duration_(sek)", "Algo2_Duration_(sek)"],
                    labels=["Algo1_012", "Algo1_120", "Algo1_210", "Algo2"],
                    title=f'{frame_data[selected_frame][1]} {options[selected_option]} run time',
                    path=f"{frame_data[selected_frame][0]}_{options[selected_option]}_run_time.png",
                    y_axis='run time (sec)'
                )
        else:
            boxplot(
                df=frame,
                columns=["Algo1_012_Duration_(sek)", "Algo1_120_Duration_(sek)", "Algo1_210_Duration_(sek)", "Algo2_Duration_(sek)"],
                labels=["Algo1_012", "Algo1_120", "Algo1_210", "Algo2"],
                title=f'{frame_data[selected_frame][1]} run time',
                path=f"{frame_data[selected_frame][0]}_run_time.png",
                y_axis='run time (sec)'
            )
