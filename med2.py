import json
import requests
from flask import Flask, request, jsonify
import logging
import re
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util

load_dotenv()
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
db = client[DB_NAME]
patient_visits_collection = db[COLLECTION_NAME]

# LLM API endpoint and authorization
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_AUTH_HEADER = os.getenv("LLM_AUTH_HEADER")

# Semantic model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Example cases for few-shot learning
EXAMPLE_CASES = {
    "diagnosis": [
        {"input": "Chief complaints: High blood pressure and chest pain.", "output": [{"code": "I10", "name": "Essential hypertension"}]},
        {"input": "Chief complaints: Persistent dry cough and mild dyspnea.", "output": [{"code": "J45.909", "name": "Allergic asthma"}]},
    ],
    "investigations": [
        {"input": "Chief complaints: High blood pressure and chest pain.", "output": "ECG, Chest X-ray, Blood tests"},
        {"input": "Chief complaints: Persistent dry cough.", "output": "CBC, CRP, spirometry"},
    ],
    "medications": [
        {"input": "Chief complaints: High blood pressure and chest pain. Medical history: Hypertension.", "output": [
            {"name": "Amlodipine", "instructions": "Take 5 mg once daily"},
            {"name": "Nitroglycerin", "instructions": "Take 0.4 mg sublingually as needed for chest pain"}
        ]},
        {"input": "Chief complaints: Sinussitis and heavy cold.", "output": [
            {"name": "Amoxicillin", "instructions": "Take 500 mg three times daily for 7 days"}
        ]},
    ],
    "followup": [
        {"input": "Chief complaints: High blood pressure and chest pain.", "output": {"date": "2025-03-12", "text": "Review in 1 week with ECG results."}},
        {"input": "Chief complaints: Persistent dry cough.", "output": {"date": "2025-03-10", "text": "Review in 2 weeks."}},
    ],
    "diet_instructions": [
        {"input": "Chief complaints: High blood pressure and chest pain.", "output": "Reduce salt intake, avoid fatty foods, increase potassium-rich foods"},
        {"input": "Chief complaints: Sinussitis and heavy cold.", "output": "Stay hydrated, avoid dairy, consume warm fluids"},
    ]
}

def format_examples(field):
    """Format examples for few-shot learning"""
    examples = EXAMPLE_CASES.get(field, [])
    return "".join([f"\nExample {i+1}:\nInput: {ex['input']}\nOutput: {json.dumps(ex['output'])}\n" for i, ex in enumerate(examples)])

def get_llm_suggestion(prompt):
    """Fetch suggestion from LLM API"""
    headers = {"Authorization": LLM_AUTH_HEADER, "Content-Type": "application/json"}
    payload = {
        "model": "unsloth/Qwen2.5-1.5B-Instruct",
        "messages": [{"role": "system", "content": "You are an expert Medical AI assistant. Provide structured outputs without explanations, matching the specified format exactly."}, {"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.2,
        "seed": 42
    }
    try:
        response = requests.post(LLM_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error calling LLM API: {str(e)}")
        return None

def get_icd11_diagnosis(diagnosis_name):
    """Fetch ICD-11 diagnosis from API"""
    try:
        url = f"https://dev.apexcura.com/api/op/getDiagnosisList?searchText={diagnosis_name}"
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        if result.get("status") and result.get("data"):
            return {"name": result["data"][0]["diagnosis"], "code": result["data"][0]["icdCode"]}
        return None
    except Exception as e:
        logger.error(f"Error fetching ICD-11 diagnosis: {str(e)}")
        return None

def fetch_patient_history(patient_id):
    """Fetch last 10 consultations for patient"""
    try:
        visits = patient_visits_collection.find({"patientId": patient_id}).sort("createdAt", -1).limit(10)
        return list(visits)
    except Exception as e:
        logger.error(f"Error fetching patient history: {str(e)}")
        return []

def fetch_doctor_preferences(doctor_id, chief_complaints):
    """Fetch doctor preferences using semantic similarity"""
    try:
        visits = patient_visits_collection.find({"doctorId": doctor_id}).sort("createdAt", -1).limit(50)
        preferences = {"diagnosis": [], "investigations": [], "medications": [], "diet_instructions": []}
        current_embedding = semantic_model.encode(chief_complaints.lower(), convert_to_tensor=True)

        for visit in visits:
            visit_complaints = visit.get("chief_complaints", {}).get("content", "").lower()
            if not visit_complaints:
                continue
            visit_embedding = semantic_model.encode(visit_complaints, convert_to_tensor=True)
            similarity = util.cos_sim(current_embedding, visit_embedding).item()
            if similarity > 0.7:  # Semantic similarity threshold
                preferences["diagnosis"].extend(visit.get("diagnosis", {}).get("list", []))
                preferences["investigations"].extend(visit.get("investigations", {}).get("list", []))
                preferences["medications"].extend(visit.get("medications", []))
                preferences["diet_instructions"].append(visit.get("diet_instructions", {}).get("content", ""))
        return preferences
    except Exception as e:
        logger.error(f"Error fetching doctor preferences: {str(e)}")
        return {}

def extract_patient_details(past_visits, data):
    """Extract patient details from past visits"""
    details = {
        "allergies": data.get("allergies", ""),
        "medical_history": data.get("medical_history", ""),
        "current_medications": data.get("current_medications", ""),
        "family_history": ""
    }
    if not past_visits:
        return details

    for visit in past_visits:
        if not details["allergies"] and visit.get("allergies", {}).get("list"):
            details["allergies"] = ", ".join([a["name"] for a in visit["allergies"]["list"]])
        if not details["medical_history"] and visit.get("medical_history", {}).get("content"):
            details["medical_history"] = visit["medical_history"]["content"]
        if not details["current_medications"] and visit.get("current_medications", {}).get("content"):
            details["current_medications"] = visit["current_medications"]["content"]
        if not details["family_history"] and visit.get("family_history", {}).get("content"):
            details["family_history"] = visit["family_history"]["content"]
        if all(details.values()):  # Stop if all fields are filled
            break
    return details

def process_llm_output(text, field):
    """Process LLM output into structured format"""
    if not text:
        return None
    json_pattern = r'\[.*\]|\{.*\}'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    
    # Fallback for non-JSON outputs
    if field == "medications" and ":" in text:
        meds = []
        for line in text.strip().split("\n"):
            if ":" in line:
                name, instr = line.split(":", 1)
                meds.append({"name": name.strip(), "instructions": instr.strip()})
        return meds if meds else None
    elif field in ["investigations", "diet_instructions"]:
        return text.strip()
    return None

def generate_prompt(field, patient_data, doctor_prefs):
    """Generate LLM prompt for a specific field"""
    base_context = f"""
Patient Information:
- Chief complaints: {patient_data.get('chief_complaints', 'Not provided')}
- Allergies: {patient_data.get('allergies', 'Not provided')}
- Medical history: {patient_data.get('medical_history', 'Not provided')}
- Current medications: {patient_data.get('current_medications', 'Not provided')}
- Vitals: {patient_data.get('vitals', 'Not provided')}
- Family history: {patient_data.get('family_history', 'Not provided')}
Doctor Preferences for similar cases:
- Diagnosis: {', '.join([d.get('name', '') for d in doctor_prefs.get('diagnosis', [])]) or 'None'}
- Investigations: {', '.join([i.get('name', '') for i in doctor_prefs.get('investigations', [])]) or 'None'}
- Medications: {', '.join([m.get('name', '') for m in doctor_prefs.get('medications', [])]) or 'None'}
- Diet Instructions: {', '.join([d for d in doctor_prefs.get('diet_instructions', []) if d]) or 'None'}
"""
    prompts = {
        "diagnosis": f"{base_context}\nProvide the most likely diagnosis based on the patient’s current complaints and history.\nOutput format: [{{\"code\": \"ICD_CODE\", \"name\": \"DIAGNOSIS_NAME\"}}]\n{format_examples('diagnosis')}",
        "investigations": f"{base_context}\nProvide a comma-separated list of investigations relevant to the patient’s condition.\nOutput format: \"test1, test2, test3\"\n{format_examples('investigations')}",
        "medications": f"{base_context}\nProvide a list of medications with dosage instructions based on the patient’s complaints and history.\nOutput format: [{{\"name\": \"MED_NAME\", \"instructions\": \"DOSAGE\"}}]\n{format_examples('medications')}",
        "followup": f"{base_context}\nProvide a followup plan based on the patient’s condition.\nOutput format: {{\"date\": \"YYYY-MM-DD\", \"text\": \"INSTRUCTIONS\"}}\n{format_examples('followup')}",
        "diet_instructions": f"{base_context}\nProvide diet instructions relevant to the patient’s condition.\nOutput format: \"INSTRUCTIONS\"\n{format_examples('diet_instructions')}"
    }
    return prompts.get(field, "")

def process_json_data(data):
    """Process input JSON and return completed data"""
    result = data.copy()
    patient_id = result.get("patientId")
    doctor_id = result.get("doctorId")
    
    # Step 1: Fetch patient history
    past_visits = fetch_patient_history(patient_id) if patient_id else []
    patient_details = extract_patient_details(past_visits, result)
    print("Patient details:", patient_details)
    result.update(patient_details)

    # Step 2: Fetch doctor preferences semantically
    doctor_prefs = fetch_doctor_preferences(doctor_id, result.get("chief_complaints", "")) if doctor_id and result.get("chief_complaints") else {}
    print("Doctor preferences:", doctor_prefs)
    # Step 3: Process fields with LLM
    fields_to_process = ["diagnosis", "investigations", "medications", "followup", "diet_instructions"]
    for field in fields_to_process:
        is_empty = (field not in result or 
                    (isinstance(result[field], list) and not result[field]) or 
                    (isinstance(result[field], str) and not result[field].strip()) or 
                    (isinstance(result[field], dict) and not any(result[field].values())))
        
        if is_empty and result.get("chief_complaints"):  # Only process if context is valid
            prompt = generate_prompt(field, result, doctor_prefs)
            llm_output = get_llm_suggestion(prompt)
            processed_output = process_llm_output(llm_output, field)
            result[field] = processed_output if processed_output is not None else (
                [] if field in ["diagnosis", "medications"] else "" if field in ["investigations", "diet_instructions"] else {}
            )

    # Step 4: Refine diagnosis with ICD-11 API
    if "diagnosis" in result and isinstance(result["diagnosis"], list) and result["diagnosis"]:
        diag_name = result["diagnosis"][0].get("name", "")
        api_diagnosis = get_icd11_diagnosis(diag_name)
        if api_diagnosis:
            result["diagnosis"] = [api_diagnosis]

    return result

@app.route('/api/suggest', methods=['POST'])
def suggest():
    """API endpoint to process input and provide suggestions"""
    try:
        data = request.json
        if not data or "patientId" not in data or "doctorId" not in data:
            return jsonify({"error": "Missing patientId or doctorId"}), 400
        completed_data = process_json_data(data)
        return jsonify(completed_data)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)