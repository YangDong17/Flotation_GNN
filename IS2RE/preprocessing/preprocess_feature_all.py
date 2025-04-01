from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets import SinglePointLmdbDataset, TrajectoryLmdbDataset
import ase.io
from ase.build import bulk
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
import matplotlib.pyplot as plt
import lmdb
import pickle
from tqdm import tqdm
import torch
import os

a2g = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy=True,    # False for test data
    r_forces=True,
    r_distances=True,
    r_fixed=True,
)

#lmdb_path = "predict_lmdb/sample_test.lmdb"

lmdb_path = "../data/feature_all_train.lmdb"

db = lmdb.open(
    lmdb_path,
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
)

def read_trajectory_extract_features(a2g, traj_path):
    traj = ase.io.read(traj_path, ":")
    tags = traj[0].get_tags()
    images = [traj[0], traj[-1]]
    data_objects = a2g.convert_all(images, disable_tqdm=True)
    data_objects[0].tags = torch.LongTensor(tags)
    data_objects[1].tags = torch.LongTensor(tags)
    return data_objects

#system_paths = ["predict_data/random712643.extxyz"]
system_paths = "../../vasprun/231213/train/"
import pandas as pd
ads_data = pd.read_csv('feature.csv')
#idx = 0
files = os.listdir(system_paths)
files.sort()
idx = 0
for system in files:
    #idx = 0
    # Extract Data object
    print(system)
    print(ads_data.iloc[int(system),1])
    #break
    raw_data = ase.io.read(os.path.join(os.path.join(system_paths,system), 'vasprun.xml'), ':')
    #tags = raw_data[0].get_tags()
    relax_data = raw_data[-1]
    for fid, data in tqdm(enumerate(raw_data), total=len(raw_data)):
        images = [data, relax_data]
        tags = images[0].get_tags()
        data_objects = a2g.convert_all(images, disable_tqdm=True)
        data_objects[0].tags = torch.LongTensor(tags)
        data_objects[1].tags = torch.LongTensor(tags)

        initial_struc = data_objects[0]
        relaxed_struc = data_objects[1]

        initial_struc.y_init = initial_struc.y # subtract off reference energy, if applicable
        del initial_struc.y
        #initial_struc.y_relaxed = relaxed_struc.y # subtract off reference energy, if applicable
        initial_struc.y_relaxed = ads_data.iloc[int(system),1]
        initial_struc.feature = [ads_data.iloc[int(system), 2], ads_data.iloc[int(system), 3],
                                 ads_data.iloc[int(system), 4]]
        initial_struc.pos_relaxed = relaxed_struc.pos

        # Filter data if necessary
        # OCP filters adsorption energies > |10| eV

        initial_struc.sid = idx  # arbitrary unique identifier

        # no neighbor edge case check
        if initial_struc.edge_index.shape[1] == 0:
            print("no neighbors", traj_path)
            continue

        # Write to LMDB
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(initial_struc, protocol=-1))
        txn.commit()
        db.sync()
        idx += 1

db.close()

dataset = SinglePointLmdbDataset({"src": lmdb_path})
print(len(dataset))
print(dataset[0])
