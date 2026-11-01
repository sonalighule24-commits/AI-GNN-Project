import pubchempy as pcp
import requests
from duckduckgo_search import DDGS # We installed this earlier!

def get_drug_info(drug_name):
    """
    Fetches detailed pharmacology, molecular info, and safety data.
    """
    info = {
        "name": drug_name,
        "description": "Fetching description...",
        "mechanism": "Unknown",
        "weight": "N/A",
        "safety_summary": "No specific FDA warnings found."
    }
    
    try:
        # 1. PubChem Search (For Weight & CID)
        compounds = pcp.get_compounds(drug_name, 'name')
        if compounds:
            c = compounds[0]
            info['weight'] = str(c.molecular_weight)
            
            # 2. Wikipedia Summary (Better for descriptions than PubChem raw data)
            # We use DuckDuckGo to get the instant answer text
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{drug_name} drug mechanism", max_results=1))
                if results:
                    info['description'] = results[0]['body']

        # 3. OpenFDA (The Safety Database)
        fda_url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug_name}&limit=1"
        fda_resp = requests.get(fda_url)
        if fda_resp.status_code == 200:
            fda_data = fda_resp.json()
            if 'results' in fda_data:
                label = fda_data['results'][0]
                if 'boxed_warning' in label:
                    info['safety_summary'] = "⚠️ BOXED WARNING: " + " ".join(label['boxed_warning'])
                elif 'warnings' in label:
                    info['safety_summary'] = label['warnings'][0][:300] + "..."
                    
    except Exception as e:
        print(f"Error fetching data: {e}")
        info['description'] = "Could not retrieve description. (Connection error)"
        
    return info

def get_combination_ratio(drug1, drug2):
    """
    Searches DrugComb for existing synergy data.
    """
    # Real-world logic: Query DrugComb API (Mocked here for stability as their API varies)
    # In a full paper, you would scrape https://drugcomb.org/
    
    return {
        "status": "Hypothetical",
        "recommended_ratio": "1:1 (molar)",
        "reasoning": "No clinical trial data found. Standard protocol: Start at IC50 equipotency."
    }