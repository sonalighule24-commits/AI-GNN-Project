import pandas as pd
import os

DATA_FOLDER = 'data'

# Load the three key files
drugs = pd.read_csv(os.path.join(DATA_FOLDER, 'biosnap_drugs.csv'))
diseases = pd.read_csv(os.path.join(DATA_FOLDER, 'biosnap_diseases.csv'))
interactions = pd.read_csv(os.path.join(DATA_FOLDER, 'biosnap_drug_disease.csv'))

print("--- DEBUGGING ID MISMATCH ---")

print("\n1. Drug IDs in Node File (biosnap_drugs.csv):")
print(drugs['drug_id'].head(3).values)

print("\n2. Disease IDs in Node File (biosnap_diseases.csv):")
print(diseases['disease_id'].head(3).values)

print("\n3. IDs in Interaction File (biosnap_drug_disease.csv):")
print(interactions.head(3))

# Check overlap
drug_overlap = set(drugs['drug_id']).intersection(set(interactions['drug_id']))
disease_overlap = set(diseases['disease_id']).intersection(set(interactions['disease_id']))

print(f"\nMatching Drugs: {len(drug_overlap)} out of {len(interactions)} interactions")
print(f"Matching Diseases: {len(disease_overlap)} out of {len(interactions)} interactions")