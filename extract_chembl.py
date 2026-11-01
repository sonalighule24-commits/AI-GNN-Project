import sqlite3
import pandas as pd
import os

# --- Configuration ---
# UPDATE THIS with the exact name of your .db file
DB_FILE = 'chembl_34.db' 
DATA_FOLDER = 'data'
DB_PATH = os.path.join(DATA_FOLDER, DB_FILE)
OUTPUT_DRUGS_PATH = os.path.join(DATA_FOLDER, 'chembl_drugs.csv')
OUTPUT_INTERACTIONS_PATH = os.path.join(DATA_FOLDER, 'chembl_drug_targets.csv')

print(f"Connecting to database at {DB_PATH}...")

try:
    conn = sqlite3.connect(DB_PATH)
    
    # --- Query 1: Extract Approved Drugs ---
    # We select drugs that have reached Phase 4 (Approved) and have a structure.
    print("Extracting approved drugs...")
    query_drugs = """
    SELECT DISTINCT
        md.chembl_id as drug_id,
        md.pref_name as name,
        cs.canonical_smiles as smiles
    FROM
        molecule_dictionary md
    JOIN
        compound_structures cs ON md.molregno = cs.molregno
    WHERE
        md.max_phase = 4
        AND cs.canonical_smiles IS NOT NULL;
    """
    
    df_drugs = pd.read_sql_query(query_drugs, conn)
    print(f"Found {len(df_drugs)} approved drugs with structures.")
    
    # Save Drugs to CSV
    df_drugs.to_csv(OUTPUT_DRUGS_PATH, index=False)
    print(f"Saved drugs to {OUTPUT_DRUGS_PATH}")

    # --- Query 2: Extract Drug-Target Interactions ---
    # We link drugs to their targets and get the UniProt ID (standard protein ID).
    print("Extracting drug-target interactions...")
    query_interactions = """
    SELECT DISTINCT
        md.chembl_id as drug_id,
        c.accession as uniprot_id,
        dm.action_type
    FROM
        molecule_dictionary md
    JOIN
        drug_mechanism dm ON md.molregno = dm.molregno
    JOIN
        target_dictionary td ON dm.tid = td.tid
    JOIN
        target_components tc ON td.tid = tc.tid
    JOIN
        component_sequences c ON tc.component_id = c.component_id
    WHERE
        md.max_phase = 4
        AND td.target_type = 'SINGLE PROTEIN';
    """
    
    df_interactions = pd.read_sql_query(query_interactions, conn)
    print(f"Found {len(df_interactions)} drug-target interactions.")
    
    # Save Interactions to CSV
    df_interactions.to_csv(OUTPUT_INTERACTIONS_PATH, index=False)
    print(f"Saved interactions to {OUTPUT_INTERACTIONS_PATH}")

    conn.close()
    print("Done!")

except sqlite3.OperationalError as e:
    print(f"Error: Could not connect to database. Make sure '{DB_FILE}' is in the 'data' folder.")
    print(f"Detailed error: {e}")