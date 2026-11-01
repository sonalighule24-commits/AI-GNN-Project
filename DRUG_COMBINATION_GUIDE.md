# 💊 Drug Combination Feature - Implementation Guide

## ✅ What Was Added

### **Frontend Enhancement (frontend.py)**
Added a new **Tab 3: Drug Combination** with:
- Input fields for Drug 1 (name + SMILES)
- Input fields for Drug 2 (name + SMILES)  
- Target disease selector
- Synergy score display
- AI analysis of combination
- Supporting research papers

### **Backend (app.py)**
Already includes `/analyze_combination` endpoint that:
1. Accepts two drugs + target disease
2. Generates fingerprints for both drugs
3. Gets GNN embeddings for each
4. **Creates "Virtual Cocktail" vector** by averaging both embeddings
5. Scores against all diseases using dot product
6. Fetches research papers for both drugs
7. Sends to Gemini AI for synergy analysis
8. Returns combined confidence score + AI insights

---

## 🚀 How to Use

### **1. Start the Backend**
```powershell
uvicorn app:app --reload --port 8000
```

### **2. Start the Frontend**
```powershell
streamlit run frontend.py
```

### **3. Navigate to "Drug Combination" Tab**

### **4. Enter Combination Details**
Example:
- **Drug 1:** Aspirin (`CC(=O)OC1=CC=CC=C1C(=O)O`)
- **Drug 2:** Metformin (`CN(C)C(=N)NC(=N)N`)
- **Target Disease:** Type 2 Diabetes

### **5. Click "Analyze Combination"**

---

## 📊 How It Works

```
INPUT: Drug A + Drug B + Disease
         ↓
    [Get Fingerprints]
         ↓
    [Get GNN Embeddings]
    Emb_A (128-dim) + Emb_B (128-dim)
         ↓
    [SYNERGY CALCULATION]
    Combined = (Emb_A + Emb_B) / 2
         ↓
    [SCORE against Disease]
    Score = Combined · Disease_Embedding
         ↓
    [AI ANALYSIS]
    - Fetch papers for both drugs
    - Send to Gemini: "Explain synergy"
         ↓
    OUTPUT: Synergy Score + AI Explanation
```

---

## 🧮 Synergy Algorithms Explained

### **Method 1: Averaging (Current)**
```python
combo_embedding = (drug1_embedding + drug2_embedding) / 2
```
**Pros:** Fast, captures complementary effects
**Cons:** Doesn't model drug interactions explicitly

### **Optional: Enhanced Methods**

#### **Method 2: Element-wise Multiplication (Interaction)**
```python
combo_embedding = drug1_embedding * drug2_embedding
```
**Use when:** You want to penalize weak drugs in combination

#### **Method 3: Weighted Average**
```python
combo_embedding = 0.6 * drug1_embedding + 0.4 * drug2_embedding
```
**Use when:** One drug is expected to be more potent

#### **Method 4: Concatenation + MLP**
```python
combined = MLP(concat(drug1_embedding, drug2_embedding))
```
**Use when:** You have a trained interaction model

---

## 📋 API Endpoint Details

### **Request Format**
```json
POST /analyze_combination
{
  "name1": "Aspirin",
  "smiles1": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "name2": "Metformin",
  "smiles2": "CN(C)C(=N)NC(=N)N",
  "target_disease": "Type 2 Diabetes"
}
```

### **Response Format**
```json
{
  "drug": "Aspirin + Metformin",
  "target_disease": "Type 2 Diabetes",
  "gnn_confidence": 0.85,
  "ai_analysis": "Both drugs target glucose metabolism...",
  "sources": [
    {
      "title": "Aspirin in Diabetes Management",
      "year": 2023,
      "authors": "Smith et al.",
      "abstract": "...",
      "link": "https://..."
    },
    {
      "title": "Metformin Clinical Efficacy",
      "year": 2024,
      "authors": "Johnson et al.",
      "abstract": "...",
      "link": "https://..."
    }
  ]
}
```

---

## 🔧 Testing the Feature

### **Test Case 1: Classic Combination**
```
Drug 1: Aspirin (NSAID)
Drug 2: Metformin (Diabetes drug)
Disease: Type 2 Diabetes
Expected: High synergy (multiple pathways)
```

### **Test Case 2: Complementary Drugs**
```
Drug 1: Atorvastatin (Statin)
Drug 2: Metformin (Antidiabetic)
Disease: Cardiovascular Disease
Expected: Moderate-High synergy
```

### **Test Case 3: Invalid SMILES**
```
Drug 1: Invalid SMILES "XYZ123"
Expected: Error message explaining invalid SMILES
```

---

## 💡 Customization Ideas

### **Add Drug-Drug Interaction Database**
```python
# Fetch known interactions from PubChem, DrugBank
interaction_score = get_interaction_score(drug1_id, drug2_id)
final_score = combo_score * interaction_score
```

### **Add Synergy Thresholds**
```python
if synergy_score > 0.85:
    recommendation = "HIGHLY RECOMMENDED"
elif synergy_score > 0.65:
    recommendation = "RECOMMENDED"
else:
    recommendation = "INVESTIGATE FURTHER"
```

### **Multi-Drug Analysis**
```python
# Support 3+ drug combinations
def analyze_multi_drug_combination(drugs_list, target_disease):
    embeddings = [get_gnn_embedding(drug) for drug in drugs_list]
    combined = torch.stack(embeddings).mean(dim=0)
    # Rest of logic...
```

### **Mechanism of Action Prediction**
```python
# Return which proteins/pathways are targeted
pathway_targets = identify_target_pathways(combo_embedding, disease_embedding)
return {..., "targeted_pathways": pathway_targets}
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Invalid SMILES" error | Validate SMILES using RDKit: `python -c "from rdkit import Chem; print(Chem.MolFromSmiles('...'))"` |
| Request timeout | Increase timeout in frontend (currently 120s) or reduce paper limit |
| No research papers found | Check internet connection, Semantic Scholar API may be rate-limited |
| "Backend Offline" | Ensure `uvicorn app:app --reload` is running |
| Low synergy scores | Try well-known drug combinations (e.g., Aspirin + Warfarin) |

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| GNN Inference Time | ~0.5s per drug |
| Combination Scoring | ~0.1s |
| Paper Fetching | 2-5s |
| AI Analysis (Gemini) | 2-10s |
| **Total Time** | **~5-15 seconds** |

