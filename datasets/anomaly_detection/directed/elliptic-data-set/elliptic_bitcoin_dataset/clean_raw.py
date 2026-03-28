
#? This file will load the elliptic dataset and save it in the pyg format with only the labeled nodes

import os
import pandas as pd
import torch
from torch_geometric.data import Data
from pathlib import Path
# Point this to your unzipped Kaggle folder
repo_root = Path.cwd().parent   # go from experiments -> repo root
root = repo_root / "datasets" / "anomaly_detection" / "directed" / "elliptic-data-set/elliptic_bitcoin_dataset"

# Typical filenames in this dataset:
# - elliptic_txs_features.csv
# - elliptic_txs_classes.csv
# - elliptic_txs_edgelist.csv
features_path = os.path.join(root, "elliptic_txs_features.csv")
classes_path  = os.path.join(root, "elliptic_txs_classes.csv")
edges_path    = os.path.join(root, "elliptic_txs_edgelist.csv")

# 1) Load
feat_df = pd.read_csv(features_path, header=None)
cls_df  = pd.read_csv(classes_path)      # columns: txId, class
edge_df = pd.read_csv(edges_path)        # columns: txId1, txId2

# 2) Fix feature column names (first col is txId, second is timestep, rest are features)
n_cols = feat_df.shape[1]
feat_df.columns = ["txId", "time_step"] + [f"f{i}" for i in range(n_cols - 2)]

# Ensure txId dtype matches across files
feat_df["txId"] = feat_df["txId"].astype(str)
cls_df["txId"]  = cls_df["txId"].astype(str)
edge_df["txId1"] = edge_df["txId1"].astype(str)
edge_df["txId2"] = edge_df["txId2"].astype(str)

# 3) Keep only labeled nodes (licit=2, illicit=1), drop unknown
cls_df = cls_df[cls_df["class"].isin(["1", "2", 1, 2])].copy()
cls_df["class"] = cls_df["class"].astype(int)

# 4) Join labels with features, keep only intersection
node_df = feat_df.merge(cls_df[["txId", "class"]], on="txId", how="inner").copy()

# Label mapping for PyG: illicit->1, licit->0 (common anomaly setup)
node_df["y"] = (node_df["class"] == 1).astype(int)

# 5) Remap txId -> [0..N-1]
node_df = node_df.reset_index(drop=True)
txid_to_idx = {txid: i for i, txid in enumerate(node_df["txId"].tolist())}

# 6) Build x and y
feature_cols = [c for c in node_df.columns if c.startswith("f")]
x = torch.tensor(node_df[feature_cols].values, dtype=torch.float)
y = torch.tensor(node_df["y"].values, dtype=torch.long)

# 7) Filter edges to kept nodes and map to indices
edge_df = edge_df[
    edge_df["txId1"].isin(txid_to_idx) & edge_df["txId2"].isin(txid_to_idx)
].copy()

src = edge_df["txId1"].map(txid_to_idx).to_numpy()
dst = edge_df["txId2"].map(txid_to_idx).to_numpy()

# Keep directed edges as-is:
edge_index = torch.tensor([src, dst], dtype=torch.long)

# (Optional) remove self-loops
mask = edge_index[0] != edge_index[1]
edge_index = edge_index[:, mask]

data = Data(x=x, edge_index=edge_index, y=y)
data.num_nodes = x.size(0)

print(data)
print("illicit count:", int((y == 1).sum()))
print("licit count:", int((y == 0).sum()))

# 8) Save in PyG format
out_path = os.path.join(root, "elliptic_labeled_directed_pyg.pt")
torch.save(data, out_path)
print("saved to:", out_path)