# File: gnn_drug_repurposing/build_graph.py
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
import pickle

print("Initializing Advanced Graph Construction (High-Res Edition)...")

DATA_FOLDER = 'data'
# IMPROVEMENT 1: Increase resolution from 128 to 1024 bits
FINGERPRINT_SIZE = 1024  

# --- Helper: InChI to Fingerprint ---
def inchi_to_fingerprint(inchi, radius=2, n_bits=FINGERPRINT_SIZE):
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is None: return np.zeros(n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return np.zeros(n_bits)

# 1. Load Data
print("Loading data...")
struct_df = pd.read_csv(os.path.join(DATA_FOLDER, 'drug_structures.csv'))
drug_target_df = pd.read_csv(os.path.join(DATA_FOLDER, 'biosnap_interactions.csv'))
ppi_df = pd.read_csv(os.path.join(DATA_FOLDER, 'biosnap_ppi.csv'))
drug_disease_df = pd.read_csv(os.path.join(DATA_FOLDER, 'biosnap_drug_disease.csv'))

# 2. Smart Mapping (Intersection)
valid_drug_ids = set(drug_disease_df['drug_id']).intersection(set(struct_df['drug_id']))
print(f"Drugs with both History and Structure: {len(valid_drug_ids)}")

struct_df = struct_df[struct_df['drug_id'].isin(valid_drug_ids)].reset_index(drop=True)
drug_disease_df = drug_disease_df[drug_disease_df['drug_id'].isin(valid_drug_ids)].reset_index(drop=True)

# ... inside build_graph.py, around Step 2 ...

# Count disease occurrences
disease_counts = drug_disease_df['disease_id'].value_counts()
print("Top 5 most common diseases:", disease_counts.head())

# Filter out diseases that appear TOO often (e.g., more than 500 times)
# These act as "Hubs" that ruin predictions
common_diseases = disease_counts[disease_counts > 500].index
print(f"Removing {len(common_diseases)} generic/hub diseases...")

drug_disease_df = drug_disease_df[~drug_disease_df['disease_id'].isin(common_diseases)]

# ... continue with creating mappings ...
# Create Mappings
drug_map = {id: i for i, id in enumerate(struct_df['drug_id'].unique())}
protein_map = {id: i for i, id in enumerate(set(drug_target_df['target_id']).union(set(ppi_df['protein_1'])).union(set(ppi_df['protein_2'])))}
disease_map = {id: i for i, id in enumerate(drug_disease_df['disease_id'].unique())}

print(f"Graph Nodes: {len(drug_map)} Drugs, {len(protein_map)} Proteins, {len(disease_map)} Diseases")

# 3. Generate Chemical Features
print(f"Generating {FINGERPRINT_SIZE}-bit Chemical Fingerprints...")
drug_features = []
for inchi in tqdm(struct_df['inchi']):
    fp = inchi_to_fingerprint(inchi)
    drug_features.append(fp)

drug_features_tensor = torch.tensor(np.array(drug_features), dtype=torch.float)

# 4. Build Graph
data = HeteroData()
data['drug'].x = drug_features_tensor

# IMPROVEMENT 2: Better initialization for unknown features
# We still use random noise for now, but we match the hidden dimension (128) for stability
data['protein'].x = torch.randn(len(protein_map), 128) 
data['disease'].x = torch.randn(len(disease_map), 128)

# Edge Helper
def get_edges(df, src_col, dst_col, src_map, dst_map):
    src, dst = [], []
    for _, row in df.iterrows():
        if row[src_col] in src_map and row[dst_col] in dst_map:
            src.append(src_map[row[src_col]])
            dst.append(dst_map[row[dst_col]])
    return torch.tensor([src, dst], dtype=torch.long)

print("Building connections...")
edge_d_dis = get_edges(drug_disease_df, 'drug_id', 'disease_id', drug_map, disease_map)
data['drug', 'treats', 'disease'].edge_index = edge_d_dis
data['disease', 'rev_treats', 'drug'].edge_index = edge_d_dis.flip(0)

edge_d_p = get_edges(drug_target_df, 'drug_id', 'target_id', drug_map, protein_map)
data['drug', 'binds', 'protein'].edge_index = edge_d_p
data['protein', 'rev_binds', 'drug'].edge_index = edge_d_p.flip(0)

edge_ppi = get_edges(ppi_df, 'protein_1', 'protein_2', protein_map, protein_map)
data['protein', 'interacts', 'protein'].edge_index = torch.cat([edge_ppi, edge_ppi.flip(0)], dim=1)

# 5. Save
print("Graph construction complete.")
torch.save(data, os.path.join(DATA_FOLDER, 'graph_data.pt'))

# Save mappings
with open(os.path.join(DATA_FOLDER, 'mappings.pkl'), 'wb') as f:
    pickle.dump({'drug_map': drug_map, 'disease_map': disease_map}, f)

print("SUCCESS! High-Resolution Graph saved.")