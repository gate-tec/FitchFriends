from fitch_graph_praktikum.nicolas.graph_io import load_relations, save_dataframe
from fitch_graph_praktikum.nicolas.generator import get_stored_cograph_relations

import pandas as pd


records = []
records_clear_e_dircet = []
for nodes in [10, 15, 20, 25, 30]:
    for x_options in range(7):
        for x_instance in range(100):

            _, all_rels = get_stored_cograph_relations(
                num_nodes=nodes, x_options=x_options, x_instances=x_instance, clear_e_direct=False
            )
            for nominal_loss, (rels, effective_loss, is_fitch_sat) in all_rels.items():
                records.append({
                    'Nodes': nodes,
                    'Sample': x_instance,
                    'x_options': x_options,
                    'Nominal_Loss': nominal_loss,
                    'Effective_Loss': effective_loss,
                    'Is_Fitch_Sat': is_fitch_sat,
                    'Relations': rels
                })

            _, all_rels = get_stored_cograph_relations(
                num_nodes=nodes, x_options=x_options, x_instances=x_instance, clear_e_direct=True
            )
            for nominal_loss, (rels, effective_loss, is_fitch_sat) in all_rels.items():
                records_clear_e_dircet.append({
                    'Nodes': nodes,
                    'Sample': x_instance,
                    'x_options': x_options,
                    'Nominal_Loss': nominal_loss,
                    'Effective_Loss': effective_loss,
                    'Is_Fitch_Sat': is_fitch_sat,
                    'Relations': rels
                })

save_dataframe('original_fitch_samples_nicolas.tsv', pd.DataFrame.from_records(records))
save_dataframe('original_fitch_samples_no_direct_nicolas.tsv', pd.DataFrame.from_records(records_clear_e_dircet))
