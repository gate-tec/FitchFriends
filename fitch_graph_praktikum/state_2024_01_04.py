# Authors:
#   - Nicolas Handke    https://github.com/gate-tec

from typing import Dict
from fitch_graph_praktikum.nicolas.generator import get_random_cograph_relations, get_stored_cograph_relations
from fitch_graph_praktikum.nicolas.graph_io import save_dataframe

import networkx as nx
import pandas as pd
from tqdm import tqdm


records: list[Dict] = []
samples_per_leafs = 100
for num_nodes, i in tqdm(
        ((num_nodes, i) for num_nodes in [10, 15, 20, 25, 30] for i in range(samples_per_leafs)),
        total=5*samples_per_leafs,
        ncols=100
):
    _, all_rels = get_random_cograph_relations(num_nodes=num_nodes, top_down=True, ensure_base_fitch=False, do_sanity_check=False)

    for nominal_loss, (rels, effective_loss, is_fitch_sat) in all_rels.items():
        records.append({
            'Nodes': num_nodes,
            'Sample': i,
            'Nominal_Loss': nominal_loss,
            'Effective_Loss': effective_loss,
            'Is_Fitch_Sat': is_fitch_sat,
            'Relations': rels
        })

frame = pd.DataFrame.from_records(records)

save_dataframe('random_samples_nicolas.tsv', frame)
