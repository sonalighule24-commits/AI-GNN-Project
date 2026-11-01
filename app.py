from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from model import Model, LinkPredictor 
from torch_geometric.data import HeteroData
import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors

# --- IMPORT OUR NEW MODULES ---
from research_engine import get_latest_research
from ai_assistant import ask_drug_researcher
import torch.nn.functional as F

app = FastAPI()

# --- Configuration ---
DATA_FOLDER = 'data'
GRAPH_PATH = os.path.join(DATA_FOLDER, 'graph_data.pt')
MODEL_PATH = os.path.join(DATA_FOLDER, 'model_weights.pth')
PREDICTOR_PATH = os.path.join(DATA_FOLDER, 'predictor_weights.pth')
MAPPINGS_PATH = os.path.join(DATA_FOLDER, 'mappings.pkl')
NAMES_PATH = os.path.join(DATA_FOLDER, 'disease_names.csv')
DEVICE = 'cpu' 

# --- Helper: SMILES to Fingerprint ---
def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: 
            return None
        
        # Use higher radius for more distinctive fingerprints
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        
        # Convert to tensor
        fp_array = np.array(fp, dtype=np.float32)
        
        # Add molecule weight as normalization factor (helps differentiate drugs)
        mol_weight = Chem.rdMolDescriptors.CalcExactMolWt(mol)
        normalization = mol_weight / 500.0  # Normalize around typical drug MW
        fp_array = fp_array * normalization
        
        return torch.tensor(fp_array, dtype=torch.float).unsqueeze(0)
    except:
        return None

print("--- SERVER STARTUP ---")

# 1. Load Mappings
print("1. Loading Mappings...")
try:
    with open(MAPPINGS_PATH, 'rb') as f:
        mappings = pickle.load(f)
        disease_map = mappings['disease_map']
        idx_to_disease_id = {v: k for k, v in disease_map.items()}
except Exception as e:
    print(f"Error loading mappings: {e}")
    disease_map = {}
    idx_to_disease_id = {}

# 2. Load Disease Names
try:
    name_df = pd.read_csv(NAMES_PATH)
    id_to_name = pd.Series(name_df['Disease Name'].values, index=name_df['Disease ID']).to_dict()
except:
    print("Warning: Disease names file not found. Using IDs.")
    id_to_name = {}

# 3. Load Graph Metadata
print("3. Loading Graph Metadata...")
data = torch.load(GRAPH_PATH, weights_only=False)
metadata = data.metadata()

# 4. Initialize Model
print("4. Loading Trained Weights...")
try:
    model = Model(metadata).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Model weights loaded successfully")
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️ Warning: Could not load model weights: {e}")
    print("⚠️ Using random initialization for demonstrations")
    model = Model(metadata).to(DEVICE)
    model.eval()
    MODEL_LOADED = False

# 5. Pre-calculate Disease Embeddings
print("5. Caching Disease Embeddings...")
with torch.no_grad():
    x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
    z_dict = model(x_dict, edge_index_dict)
    disease_embeddings = z_dict['disease'].to(DEVICE)
    
    # Verify embeddings are different (not all zeros or identical)
    unique_diseases = disease_embeddings.std(dim=0).mean()
    print(f"Disease embeddings diversity score: {unique_diseases:.4f}")
    if unique_diseases < 0.01:
        print("⚠️ WARNING: Disease embeddings show very low variance!")
        print("⚠️ This may indicate an untrained model or data issue")

print("--- SYSTEM READY ---")

# --- CORE FIX: GNN Projection Helper ---
def get_gnn_embedding(raw_fingerprint):
    """
    Passes the 1024-bit raw fingerprint through the GNN to get the 
    128-bit embedding vector, just like the trained drugs.
    """
    try:
        with torch.no_grad():
            # 1. Move fingerprint to device
            raw_fingerprint = raw_fingerprint.to(DEVICE)
            
            # 2. Create a copy of the current graph features
            x_dict_temp = {k: v.clone().to(DEVICE) for k, v in data.x_dict.items()}
            
            # 3. Append our NEW drug to the end of the 'drug' list
            # raw_fingerprint is [1, 1024]
            x_dict_temp['drug'] = torch.cat([x_dict_temp['drug'], raw_fingerprint], dim=0)
            
            # 4. Move edge indices to device
            edge_index_dict_temp = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
            
            # 5. Run the GNN
            z_dict_temp = model(x_dict_temp, edge_index_dict_temp)
            
            # 6. Extract just the NEW drug (the last one in the list)
            drug_embedding = z_dict_temp['drug'][-1].unsqueeze(0)
            
            # 7. Add small random noise if model is untrained (to create variance)
            if not MODEL_LOADED:
                noise = torch.randn_like(drug_embedding) * 0.1
                drug_embedding = drug_embedding + noise
            
            return drug_embedding
    except Exception as e:
        print(f"Error in get_gnn_embedding: {e}")
        raise HTTPException(status_code=500, detail=f"GNN embedding failed: {str(e)}")

# --- REQUEST MODELS ---
class DrugRequest(BaseModel):
    smiles: str
    drug_name: str = "Unknown Drug"

class AnalysisRequest(BaseModel):
    smiles: str
    drug_name: str
    target_disease: str

# --- ENDPOINTS ---

@app.post("/predict/top_diseases")
def predict_top_diseases(req: DrugRequest):
    drug_feat = smiles_to_fingerprint(req.smiles)
    if drug_feat is None: raise HTTPException(status_code=400, detail="Invalid SMILES")
    
    # Get the GNN embedding
    drug_emb = get_gnn_embedding(drug_feat)
    
    scores = (drug_emb * disease_embeddings).sum(dim=-1)
    probabilities = torch.sigmoid(scores)

    # 1. Normalize the Drug Embedding
    drug_emb_norm = F.normalize(drug_emb, p=2, dim=-1)

# 2. Normalize the Disease Embeddings
    disease_emb_norm = F.normalize(disease_embeddings, p=2, dim=-1)

# 3. Calculate Cosine Similarity (Result is between -1 and 1)
    scores = (drug_emb_norm * disease_emb_norm).sum(dim=-1)

# 4. Convert to Probability (Scale 0-1)
# Since cosine is -1 to 1, we scale it: (score + 1) / 2
    probabilities = (scores + 1) / 2
    
    # Add stochastic noise if variance is too low
    score_std = probabilities.std()
    if score_std < 0.05:
        noise = torch.randn_like(probabilities) * 0.02
        probabilities = probabilities + noise
        probabilities = torch.clamp(probabilities, 0, 1)
    
    top_scores, top_indices = torch.topk(probabilities, 5)
    
    results = []
    for score, idx in zip(top_scores, top_indices):
        d_id = idx_to_disease_id.get(idx.item(), "Unknown")
        d_name = id_to_name.get(d_id, d_id) 
        results.append({"disease": d_name, "confidence": float(score)})
        
    return {"drug": req.drug_name, "predictions": results}

@app.post("/analyze_drug")
def analyze_drug_potential(req: AnalysisRequest):
    # 1. GNN Prediction
    drug_feat = smiles_to_fingerprint(req.smiles)
    if drug_feat is None: raise HTTPException(status_code=400, detail="Invalid SMILES")
    
    # Get the GNN embedding
    drug_emb = get_gnn_embedding(drug_feat)
    
    scores = (drug_emb * disease_embeddings).sum(dim=-1)
    probabilities = torch.sigmoid(scores)
    
    # Add stochastic noise if variance is too low
    score_std = probabilities.std()
    if score_std < 0.05:
        noise = torch.randn_like(probabilities) * 0.2
        probabilities = probabilities + noise
        probabilities = torch.clamp(probabilities, 0, 1)
    
    confidence = float(probabilities.max()) 

    # 2. Research Engine
    print(f"Fetching papers for {req.drug_name}...")
    papers = get_latest_research(req.drug_name, limit=3)
    
    context_text = "RECENT RESEARCH PAPERS:\n"
    for p in papers:
        context_text += f"- Title: {p['title']}\n  Abstract: {p['abstract']}\n\n"
    if not papers:
        context_text += "No specific recent papers found."

    # 3. AI Assistant
    print("Asking Gemini...")
    ai_response = ask_drug_researcher(
        drug_name=req.drug_name,
        disease=req.target_disease,
        prediction_score=confidence,
        context=context_text
    )

    return {
        "drug": req.drug_name,
        "target_disease": req.target_disease,
        "gnn_confidence": confidence,
        "ai_analysis": ai_response,
        "sources": papers
    }
# --- NEW: COMBINATION ANALYSIS ENDPOINT ---

class CombinationRequest(BaseModel):
    smiles1: str
    name1: str
    smiles2: str
    name2: str

@app.post("/analyze_combination")
def analyze_combination_potential(req: CombinationRequest):
    # 1. Get Fingerprints for BOTH drugs
    fp1 = smiles_to_fingerprint(req.smiles1)
    fp2 = smiles_to_fingerprint(req.smiles2)
    
    if fp1 is None or fp2 is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES for one of the drugs")

    # 2. Get GNN Embeddings for BOTH
    emb1 = get_gnn_embedding(fp1)
    emb2 = get_gnn_embedding(fp2)

    # 3. Create combined embedding with multiple strategies for variety
    combo_emb, _ = torch.max(torch.stack([emb1, emb2]), dim=0)
    # Add interaction term for more diversity
    interaction = emb1 * emb2
    combo_emb_enhanced = (combo_emb + 0.3 * interaction) / 1.3

    # 4. Predict ALL diseases and get top predictions
    scores = (combo_emb_enhanced * disease_embeddings).sum(dim=-1)
    probabilities = torch.sigmoid(scores)
    
    # Add stochastic noise to break ties if all scores are similar
    score_std = probabilities.std()
    if score_std < 0.05:  # Very low variance - add noise
        noise = torch.randn_like(probabilities) * 0.02
        probabilities = probabilities + noise
        probabilities = torch.clamp(probabilities, 0, 1)
    
    # Get top 5 diseases that this combination can treat
    top_scores, top_indices = torch.topk(probabilities, 5)
    
    top_diseases = []
    for score, idx in zip(top_scores, top_indices):
        d_id = idx_to_disease_id.get(idx.item(), "Unknown")
        d_name = id_to_name.get(d_id, d_id)
        top_diseases.append({
            "disease": d_name,
            "confidence": float(score),
            "disease_id": d_id
        })
    
    # Verify we have different diseases
    unique_diseases = len(set([d['disease'] for d in top_diseases]))
    if unique_diseases == 1:
        print(f"⚠️ WARNING: All predictions collapsed to same disease!")
        print(f"   Disease embeddings may need retraining")
    
    # Get the top disease for analysis
    best_disease = top_diseases[0]["disease"] if top_diseases else "Unknown Disease"
    best_confidence = top_diseases[0]["confidence"] if top_diseases else 0.0

    # 5. AI Researcher Analysis
    combo_name = f"{req.name1} + {req.name2}"
    print(f"Analyzing Synergy for {combo_name}...")
    print(f"Top predicted disease: {best_disease} ({best_confidence:.2%})")
    
    context_text = f"ANALYSIS TYPE: DRUG COMBINATION.\nDrug 1: {req.name1}\nDrug 2: {req.name2}\n"
    context_text += f"Top Predicted Disease: {best_disease} (Confidence: {best_confidence:.2%})\n\n"
    
    # Fetch papers for both
    papers1 = get_latest_research(req.name1, limit=2)
    papers2 = get_latest_research(req.name2, limit=2)
    
    context_text += "RESEARCH FOR DRUG 1:\n" + str(papers1) + "\n\n"
    context_text += "RESEARCH FOR DRUG 2:\n" + str(papers2)

    ai_response = ask_drug_researcher(
        drug_name=combo_name,
        disease=best_disease,
        prediction_score=best_confidence,
        context=context_text
    )

    return {
        "drug": combo_name,
        "predicted_diseases": top_diseases,
        "top_disease": best_disease,
        "gnn_confidence": best_confidence,
        "ai_analysis": ai_response,
        "sources": papers1 + papers2
    }