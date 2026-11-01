from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import google.generativeai as genai
import json
import os

# Install: pip install fastapi uvicorn google-generativeai

app = FastAPI(title="Drug Repurposing API")

# 1. DATA MODELS (Same as your Streamlit app)
class Citation(BaseModel):
    title: str
    year: str

class PathwayStep(BaseModel):
    step_number: int
    description: str

class MechanismReport(BaseModel):
    drug_name: str
    target_disease: str
    primary_protein_target: str
    biological_mechanism: str
    detailed_pathway: List[PathwayStep]
    predicted_efficacy_score: int
    key_citations: List[Citation]

class AnalysisRequest(BaseModel):
    drug_name: str
    disease_name: str
    api_key: str  # In production, store this in env variables, don't pass it in body

# 2. THE ENDPOINT
@app.post("/analyze", response_model=MechanismReport)
async def analyze_drug(request: AnalysisRequest):
    try:
        # Configure Gemini
        genai.configure(api_key=request.api_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": MechanismReport
            }
        )

        prompt = f"""
        Analyze the repurposing potential of {request.drug_name} for {request.disease_name}.
        Identify the protein target, detailed pathway, and provide verifiable citations.
        """

        response = model.generate_content(prompt)
        
        # Parse and return
        return json.loads(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. HOW TO RUN
# Command: uvicorn api_service:app --reload
# Your main website can now fetch: POST http://localhost:8000/analyze