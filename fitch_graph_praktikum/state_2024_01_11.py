from fitch_graph_praktikum.nicolas.graph_io import load_dataframe, save_dataframe
from fitch_graph_praktikum.alex.WP1.benchmarking import benchmark_algorithms_on_all_samples


paths = [
    "original_fitch_samples_nicolas.tsv",
    # "original_fitch_samples_no_direct_nicolas.tsv",
    # "random_fitch_samples_nicolas.tsv",
    # "random_fitch_samples_no_direct_nicolas.tsv"
]

for path in paths:
    dataset = load_dataframe(path)

    bm_dataset = benchmark_algorithms_on_all_samples(dataset)

    save_dataframe(f"bm_{path}", bm_dataset)
