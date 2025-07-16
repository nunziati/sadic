from biopandas.pdb import PandasPdb as pd
from biopandas.pdb.pandas_pdb import PandasPdb

import numpy as np
import os

pdb_code = "1ay3"

entity = pd().fetch_pdb(pdb_code)

depth_index = np.load(os.path.join("/repo/nunziati/sadic/dev/new_experiments/aligned_spacefill_translated_sphere_0.5/depth_indexes", f"{pdb_code}.npy"))

entity.df["ATOM"] = entity.df["ATOM"][entity.df["ATOM"]["alt_loc"].isin(["", "A"])]

print(entity.df["ATOM"][["record_name", "atom_number", "b_factor"]].head())
print(depth_index[:10])
print(depth_index.min(), depth_index.max(), depth_index.mean())
print(entity.df["ATOM"])
entity.df["ATOM"]["b_factor"] = 2 - depth_index
print(entity.df["ATOM"][["record_name", "atom_number", "b_factor"]].head())

entity.to_pdb(f"/repo/nunziati/sadic/dev/{pdb_code}.pdb")