from pyrfu import mms
from mms_codes import epoch_analysis
import pandas as pd
import pickle

mms.db_init(default="local", local="/data/mms")

tints = pd.read_csv("tints_dfs_brst.csv", index_col=0)
tints = [[tints.loc[i, "0"], tints.loc[i, "1"]] for i in range(len(tints))]

window_ti = 10
ss = "b"
xs_cond = None

dict_ts_all = epoch_analysis(tints, window_ti = window_ti, 
                             scale_with= ss, roll_t=30,
                             roll=1, roll_f=30, roll_feeps=None,
                             do_fit=False,
                             energies_fpi=[1, 4, 9, 16, 28],
                             energy_ranges_feeps=[[32.1, 50], [50, 100], [102, 130]],
                             fit_path = f"fit_dict_{len(tints)}DFs.pickle", xs_cond = xs_cond)

if xs_cond is not None:
    with open(f"epoch_analysis_db_{window_ti}{ss}_{len(tints)}DFs_Xs0.pickle", 'wb') as file:
        pickle.dump(dict_ts_all[0], file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"epoch_analysis_db_{window_ti}{ss}_{len(tints)}DFs_Xs1.pickle", 'wb') as file:
        pickle.dump(dict_ts_all[1], file, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f"epoch_analysis_db_{window_ti}{ss}_{len(tints)}DFs.pickle", 'wb') as file:
        pickle.dump(dict_ts_all, file, protocol=pickle.HIGHEST_PROTOCOL)