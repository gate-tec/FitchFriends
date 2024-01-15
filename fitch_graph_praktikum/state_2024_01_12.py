from fitch_graph_praktikum.nicolas.graph_io import save_dataframe
from fitch_graph_praktikum.nicolas.benchmark_WP2.benchmarking import benchmark_all_stored_graphs


tuples = [
    (0.7, 0.3),
    (0.7, 0.4),
    (0.7, 0.5)
]

for i, (mu_TP, mu_FP) in enumerate(tuples):
    print(f"#####\nRunning benchmark {i + 1}/{len(tuples)}\n#####")
    benchmark_df, weight_df = benchmark_all_stored_graphs(mu_TP=mu_TP, mu_FP=mu_FP, safety_steps=500)

    save_dataframe(f'bm2_{mu_TP}_{mu_FP}_results.tsv', benchmark_df)
    save_dataframe(f'bm2_{mu_TP}_{mu_FP}_weights.tsv', benchmark_df)
