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
import certifi
from bson import ObjectId 
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
        visits = patient_visits_collection.find({"patientId": ObjectId(patient_id)}).sort("createdAt", -1).limit(10)

        history_list = list(visits)
        logger.info(f"Fetched {len(history_list)} past consultation records for patient {patient_id}")
        return history_list
    except Exception as e:
        logger.error(f"Error fetching patient history: {str(e)}")
        return []

def fetch_doctor_preferences(doctor_id, chief_complaints):
    """Fetch doctor preferences using semantic similarity"""
    if not chief_complaints or not isinstance(chief_complaints, str) or not chief_complaints.strip():
        logger.info("No valid chief complaints provided for semantic search")
        return {}
    
    try:
        # visits = patient_visits_collection.find({"doctorId": doctor_id}).sort("createdAt", -1).limit(50)
        visits = patient_visits_collection.find({"doctorId": ObjectId(doctor_id)}).sort("createdAt", -1).limit(50)
        doctor_visits = list(visits)
        logger.info(f"Fetched {len(doctor_visits)} past consultation records for doctor {doctor_id}")
        
        preferences = {"diagnosis": [], "investigations": [], "medications": [], "diet_instructions": []}
        
        current_embedding = semantic_model.encode(chief_complaints.lower(), convert_to_tensor=True)
        
        similar_visits_found = 0
        for visit in doctor_visits:
            # Handle different formats of chief_complaints
            visit_complaints = ""
            if "chief_complaints" in visit:
                if isinstance(visit["chief_complaints"], dict):
                    visit_complaints = visit["chief_complaints"].get("content", "").lower()
                elif isinstance(visit["chief_complaints"], str):
                    visit_complaints = visit["chief_complaints"].lower()
            
            if not visit_complaints:
                continue
                
            visit_embedding = semantic_model.encode(visit_complaints, convert_to_tensor=True)
            similarity = util.cos_sim(current_embedding, visit_embedding).item()
            
            if similarity > 0.4:  # Semantic similarity threshold
                similar_visits_found += 1
                logger.info(f"Found similar visit with score {similarity:.3f}: '{visit_complaints}'")
                
                # Add diagnosis
                if "diagnosis" in visit:
                    if isinstance(visit["diagnosis"], dict) and visit["diagnosis"].get("list"):
                        preferences["diagnosis"].extend(visit["diagnosis"].get("list", []))
                
                # Add investigations
                if "investigations" in visit:
                    if isinstance(visit["investigations"], dict):
                        if visit["investigations"].get("list"):
                            preferences["investigations"].extend(visit["investigations"].get("list", []))
                        elif visit["investigations"].get("content"):
                            content = visit["investigations"].get("content")
                            if content:
                                preferences["investigations"].append({"name": content})
                
                # Add medications
                if "medications" in visit and visit.get("medications"):
                    preferences["medications"].extend(visit.get("medications", []))
                
                # Add diet instructions
                if "diet_instructions" in visit:
                    if isinstance(visit["diet_instructions"], dict) and visit["diet_instructions"].get("content"):
                        content = visit["diet_instructions"].get("content", "")
                        if content:
                            preferences["diet_instructions"].append(content)
                    elif isinstance(visit["diet_instructions"], str) and visit["diet_instructions"]:
                        preferences["diet_instructions"].append(visit["diet_instructions"])
        
        logger.info(f"Found {similar_visits_found} semantically similar visits")
        logger.info(f"Collected preferences: diagnosis ({len(preferences['diagnosis'])}), "
                   f"investigations ({len(preferences['investigations'])}), "
                   f"medications ({len(preferences['medications'])}), "
                   f"diet instructions ({len(preferences['diet_instructions'])})")
        
        return preferences
    except Exception as e:
        logger.error(f"Error fetching doctor preferences: {str(e)}")
        return {}
def extract_patient_details(past_visits, data):
    """Extract patient details from past visits"""
    # Initialize with any existing data
    allergies = ""
    medical_history = ""
    current_medications = ""
    family_history = ""
    
    # Handle existing data in various formats
    if "allergies" in data:
        if isinstance(data["allergies"], dict) and data["allergies"].get("content"):
            allergies = data["allergies"].get("content", "")
        elif isinstance(data["allergies"], str):
            allergies = data["allergies"]
    
    if "medical_history" in data:
        if isinstance(data["medical_history"], dict) and data["medical_history"].get("content"):
            medical_history = data["medical_history"].get("content", "")
        elif isinstance(data["medical_history"], str):
            medical_history = data["medical_history"]
            
    if "current_medications" in data:
        if isinstance(data["current_medications"], dict) and data["current_medications"].get("content"):
            current_medications = data["current_medications"].get("content", "")
        elif isinstance(data["current_medications"], str):
            current_medications = data["current_medications"]
            
    if "family_history" in data:
        if isinstance(data["family_history"], dict) and data["family_history"].get("content"):
            family_history = data["family_history"].get("content", "")
        elif isinstance(data["family_history"], str):
            family_history = data["family_history"]
    
    details = {
        "allergies": allergies,
        "medical_history": medical_history,
        "current_medications": current_medications,
        "family_history": family_history
    }
    
    if not past_visits:
        logger.info("No past visits found for patient details extraction")
        return details

    found_fields = set()
    logger.info(f"Extracting patient details from {len(past_visits)} past visits")
    
    for visit in past_visits:
        # Only fill in fields that aren't already populated
        
        # Check for allergies
        if "allergies" not in found_fields and not details["allergies"]:
            if "allergies" in visit:
                if isinstance(visit["allergies"], dict):
                    if visit["allergies"].get("list"):
                        details["allergies"] = ", ".join([a["name"] for a in visit["allergies"]["list"] if "name" in a])
                        found_fields.add("allergies")
                        logger.info(f"Found allergies (list): {details['allergies']}")
                    elif visit["allergies"].get("content"):
                        details["allergies"] = visit["allergies"]["content"]
                        found_fields.add("allergies")
                        logger.info(f"Found allergies (content): {details['allergies']}")
                elif isinstance(visit["allergies"], str) and visit["allergies"]:
                    details["allergies"] = visit["allergies"]
                    found_fields.add("allergies")
                    logger.info(f"Found allergies (string): {details['allergies']}")
        
        # Check for medical history
        if "medical_history" not in found_fields and not details["medical_history"]:
            if "medical_history" in visit:
                if isinstance(visit["medical_history"], dict) and visit["medical_history"].get("content"):
                    details["medical_history"] = visit["medical_history"]["content"]
                    found_fields.add("medical_history")
                    logger.info(f"Found medical history: {details['medical_history']}")
                elif isinstance(visit["medical_history"], str) and visit["medical_history"]:
                    details["medical_history"] = visit["medical_history"]
                    found_fields.add("medical_history")
                    logger.info(f"Found medical history (string): {details['medical_history']}")
        
        # Check for current medications
        if "current_medications" not in found_fields and not details["current_medications"]:
            if "current_medications" in visit:
                if isinstance(visit["current_medications"], dict) and visit["current_medications"].get("content"):
                    details["current_medications"] = visit["current_medications"]["content"]
                    found_fields.add("current_medications")
                    logger.info(f"Found current medications: {details['current_medications']}")
                elif isinstance(visit["current_medications"], str) and visit["current_medications"]:
                    details["current_medications"] = visit["current_medications"]
                    found_fields.add("current_medications")
                    logger.info(f"Found current medications (string): {details['current_medications']}")
        
        # Check for family history
        if "family_history" not in found_fields and not details["family_history"]:
            if "family_history" in visit:
                if isinstance(visit["family_history"], dict) and visit["family_history"].get("content"):
                    details["family_history"] = visit["family_history"]["content"]
                    found_fields.add("family_history")
                    logger.info(f"Found family history: {details['family_history']}")
                elif isinstance(visit["family_history"], str) and visit["family_history"]:
                    details["family_history"] = visit["family_history"]
                    found_fields.add("family_history")
                    logger.info(f"Found family history (string): {details['family_history']}")
        
        if len(found_fields) == 4:  # All fields found
            break
            
    return details

def process_llm_output(text, field):
    """Process LLM output into structured format"""
    if not text:
        logger.warning(f"No text received for field {field}")
        return None
        
    # Try to find JSON in the text
    json_pattern = r'\[.*\]|\{.*\}'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
    
    # Fallback for non-JSON outputs
    logger.info(f"Falling back to non-JSON processing for {field}")
    
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
    # Extract chief complaints (handle both string and dict formats)
    chief_complaints = ""
    if "chief_complaints" in patient_data:
        if isinstance(patient_data["chief_complaints"], dict):
            chief_complaints = patient_data["chief_complaints"].get("content", "")
        elif isinstance(patient_data["chief_complaints"], str):
            chief_complaints = patient_data["chief_complaints"]
    
    # Extract allergies (handle both string and dict formats)
    allergies = ""
    if "allergies" in patient_data:
        if isinstance(patient_data["allergies"], dict):
            allergies = patient_data["allergies"].get("content", "")
        elif isinstance(patient_data["allergies"], str):
            allergies = patient_data["allergies"]
    
    # Extract medical history (handle both string and dict formats)
    medical_history = ""
    if "medical_history" in patient_data:
        if isinstance(patient_data["medical_history"], dict):
            medical_history = patient_data["medical_history"].get("content", "")
        elif isinstance(patient_data["medical_history"], str):
            medical_history = patient_data["medical_history"]
    
    # Extract current medications (handle both string and dict formats)
    current_medications = ""
    if "current_medications" in patient_data:
        if isinstance(patient_data["current_medications"], dict):
            current_medications = patient_data["current_medications"].get("content", "")
        elif isinstance(patient_data["current_medications"], str):
            current_medications = patient_data["current_medications"]
    
    # Extract family history (handle both string and dict formats)
    family_history = ""
    if "family_history" in patient_data:
        if isinstance(patient_data["family_history"], dict):
            family_history = patient_data["family_history"].get("content", "")
        elif isinstance(patient_data["family_history"], str):
            family_history = patient_data["family_history"]
    
    # Extract vitals
    vitals = patient_data.get("vitals_string", "") or "Not provided"
    
    base_context = f"""
Patient Information:
- Chief complaints: {chief_complaints or "Not provided"}
- Allergies: {allergies or "Not provided"}
- Medical history: {medical_history or "Not provided"}
- Current medications: {current_medications or "Not provided"}
- Vitals: {vitals}
- Family history: {family_history or "Not provided"}
"""

    # Add doctor's preferred treatments for similar cases if available
    if doctor_prefs:
        base_context += "Doctor Preferences for similar cases:\n"
        
        if doctor_prefs.get('diagnosis'):
            diagnoses = [d.get('name', '') for d in doctor_prefs.get('diagnosis', []) if d.get('name')]
            if diagnoses:
                base_context += f"- Previous diagnoses: {', '.join(diagnoses[:3])}\n"
                
        if doctor_prefs.get('investigations'):
            investigations = [i.get('name', '') for i in doctor_prefs.get('investigations', []) if i.get('name')]
            if investigations:
                base_context += f"- Previous investigations: {', '.join(investigations[:5])}\n"
                
        if doctor_prefs.get('medications'):
            medications = [m.get('name', '') for m in doctor_prefs.get('medications', []) if m.get('name')]
            if medications:
                base_context += f"- Previous medications: {', '.join(medications[:5])}\n"
                
        if doctor_prefs.get('diet_instructions'):
            diet_samples = [d for d in doctor_prefs.get('diet_instructions', []) if d][:2]
            if diet_samples:
                base_context += f"- Previous diet instructions: {diet_samples[0][:50]}...\n"

    prompts = {
        "diagnosis": f"{base_context}\nProvide the most likely diagnosis based on the patient's current complaints and history.\nOutput format: [{{\"code\": \"ICD_CODE\", \"name\": \"DIAGNOSIS_NAME\"}}]\n{format_examples('diagnosis')}",
        
        "investigations": f"{base_context}\nProvide a comma-separated list of investigations relevant to the patient's condition.\nOutput format: \"test1, test2, test3\"\n{format_examples('investigations')}",
        
        "medications": f"{base_context}\nProvide a list of medications with dosage instructions based on the patient's complaints and history.\nOutput format: [{{\"name\": \"MED_NAME\", \"instructions\": \"DOSAGE\"}}]\n{format_examples('medications')}",
        
        "followup": f"{base_context}\nProvide a followup plan based on the patient's condition.\nOutput format: {{\"date\": \"YYYY-MM-DD\", \"text\": \"INSTRUCTIONS\"}}\n{format_examples('followup')}",
        
        "diet_instructions": f"{base_context}\nProvide diet instructions relevant to the patient's condition.\nOutput format: \"INSTRUCTIONS\"\n{format_examples('diet_instructions')}"
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
    logger.info(f"Patient details extracted: {patient_details}")
    result.update(patient_details)

    # Step 2: Handle chief_complaints properly (could be string or dict)
    chief_complaints_content = ""
    if "chief_complaints" in result:
        if isinstance(result["chief_complaints"], dict):
            chief_complaints_content = result["chief_complaints"].get("content", "")
        elif isinstance(result["chief_complaints"], str):
            chief_complaints_content = result["chief_complaints"]
            # Convert string to proper format
            result["chief_complaints"] = {"content": chief_complaints_content, "list": []}
    
    logger.info(f"Chief complaints content: '{chief_complaints_content}'")
    
    # Step 3: Fetch doctor preferences semantically
    doctor_prefs = {}
    if doctor_id and chief_complaints_content:
        doctor_prefs = fetch_doctor_preferences(doctor_id, chief_complaints_content)
        logger.info("Doctor preferences fetched based on semantic search")
    
    # Step 4: Process fields with LLM
    fields_to_process = ["diagnosis", "investigations", "medications", "followup", "diet_instructions"]
    for field in fields_to_process:
        is_empty = (field not in result or 
                    (isinstance(result.get(field), list) and not result[field]) or 
                    (isinstance(result.get(field), str) and not result[field].strip()) or 
                    (isinstance(result.get(field), dict) and not any(result[field].values() if result[field] else [])))
        
        # Only process if we have chief complaints (context is valid)
        if is_empty and chief_complaints_content:
            logger.info(f"Generating {field} with LLM")
            prompt = generate_prompt(field, result, doctor_prefs)
            llm_output = get_llm_suggestion(prompt)
            
            if llm_output:
                logger.info(f"Received LLM response for {field}")
                processed_output = process_llm_output(llm_output, field)
                
                if processed_output is not None:
                    result[field] = processed_output
                    logger.info(f"Set {field} to: {processed_output}")
                else:
                    logger.warning(f"Failed to process LLM output for {field}")
            else:
                logger.warning(f"No LLM output received for {field}")

    # Step 5: Refine diagnosis with ICD-11 API
    if "diagnosis" in result and isinstance(result.get("diagnosis"), list) and result["diagnosis"]:
        diag_name = result["diagnosis"][0].get("name", "")
        logger.info(f"Refining diagnosis: {diag_name}")
        api_diagnosis = get_icd11_diagnosis(diag_name)
        if api_diagnosis:
            result["diagnosis"] = [api_diagnosis]
            logger.info(f"Updated diagnosis to: {api_diagnosis}")

    return result

@app.route('/api/suggest', methods=['POST'])
def suggest():
    """API endpoint to process input and provide suggestions"""
    try:
        data = request.json
        if not data or "patientId" not in data or "doctorId" not in data:
            logger.error("Missing patientId or doctorId in request")
            return jsonify({"error": "Missing patientId or doctorId"}), 400
        
        # Log the input data for debugging
        logger.info(f"Processing request for patient {data.get('patientId')} and doctor {data.get('doctorId')}")
        logger.info(f"Input data structure: {json.dumps({k: type(v).__name__ for k, v in data.items()})}")
        
        # Check if chief_complaints exists and log its format
        if "chief_complaints" in data:
            if isinstance(data["chief_complaints"], dict):
                logger.info(f"Chief complaints (dict): {data['chief_complaints'].get('content', '')}")
            else:
                logger.info(f"Chief complaints (other type): {data['chief_complaints']}")
        else:
            logger.warning("No chief_complaints field in request data")
            
        completed_data = process_json_data(data)
        return jsonify(completed_data)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)