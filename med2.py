from dotenv import load_dotenv
load_dotenv()
import json
import requests
from flask import Flask, request, jsonify
import logging
import re
import os
from pymongo import MongoClient
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util
import certifi
from bson import ObjectId


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
        {"input": "Chief complaints: cold and ingestion", "output": [{"code": "J00", "name": "Acute nasopharyngitis (common cold)"}]},
        {"input": "Chief complaints: rash and itching", "output": [{"code": "L50.9", "name": "Urticaria, unspecified"}]},
        {"input": "Chief complaints: Ae JCB hit the patient on ribs area in the previous xrays primary xray shows there are no fractures and cracks in the ribs but the patient is feeling severe pain.while moving his hand ,inhaling,exhaling he is not able to sleep on any side.", "output": [{"code": "S29.9", "name": "Unspecified injury of thorax"}]},
    ],
    "investigations": [
        {"input": "Chief complaints: High blood pressure and chest pain.", "output": "ECG, Chest X-ray, Blood tests"},
        {"input": "Chief complaints: cold and ingestion", "output": "CBC, Chest X-ray, Allergy testing"},
        {"input": "Chief complaints: rash and itching", "output": "CBC, Skin allergy test, IgE levels"},
        {"input": "Chief complaints: Ae JCB hit the patient on ribs area in the previous xrays primary xray shows there are no fractures and cracks in the ribs but the patient is feeling severe pain.while moving his hand ,inhaling,exhaling he is not able to sleep on any side.", "output": "Chest CT, Repeat X-ray, Ultrasound of chest"},
    ],
    "medications": [
        {"input": "Chief complaints: High blood pressure and chest pain. Medical history: Hypertension.", "output": [
            {"name": "Amlodipine", "instructions": "Take 5 mg once daily"},
            {"name": "Nitroglycerin", "instructions": "Take 0.4 mg sublingually as needed for chest pain"}
        ]},
        {"input": "Chief complaints: Sinussitis and heavy cold.", "output": [
            {"name": "Amoxicillin", "instructions": "Take 500 mg three times daily for 7 days"}
        ]},
        {"input": "Chief complaints: Ae JCB hit the patient on ribs area in the previous xrays primary xray shows there are no fractures and cracks in the ribs but the patient is feeling severe pain.while moving his hand ,inhaling,exhaling he is not able to sleep on any side.", "output": [
            {"name": "Ibuprofen", "instructions": "Take 400 mg every 6 hours as needed for pain"},
            {"name": "Acetaminophen", "instructions": "Take 500 mg every 6 hours as needed for pain"}
        ]},
    ],
    "followup": [
        {"input": "Chief complaints: High blood pressure and chest pain.", "output": {"date": "2025-03-12", "text": "Review in 1 week with ECG results."}},
        {"input": "Chief complaints: cold and ingestion", "output": {"date": "2025-03-10", "text": "Review in 5 days."}},
        {"input": "Chief complaints: Ae JCB hit the patient on ribs area in the previous xrays primary xray shows there are no fractures and cracks in the ribs but the patient is feeling severe pain.while moving his hand ,inhaling,exhaling he is not able to sleep on any side.", "output": {"date": "2025-03-13", "text": "Review in 1 week with imaging results."}},
    ],
    "diet_instructions": [
        {"input": "Chief complaints: High blood pressure and chest pain.", "output": "Reduce salt intake, avoid fatty foods, increase potassium-rich foods"},
        {"input": "Chief complaints: cold and ingestion", "output": "Stay hydrated, drink plenty of fluids, avoid spicy foods"},
        {"input": "Chief complaints: Ae JCB hit the patient on ribs area in the previous xrays primary xray shows there are no fractures and cracks in the ribs but the patient is feeling severe pain.while moving his hand ,inhaling,exhaling he is not able to sleep on any side.", "output": "Eat soft foods, avoid heavy lifting, maintain light diet"}
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
        url = f"https://dev.apexcura.com/api/op/getDiagnosisList?search={diagnosis_name}"
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
        visits = patient_visits_collection.find({"doctorId": ObjectId(doctor_id)}).sort("createdAt", -1).limit(50)
        doctor_visits = list(visits)
        logger.info(f"Fetched {len(doctor_visits)} past consultation records for doctor {doctor_id}")
        
        preferences = {"diagnosis": [], "investigations": [], "medications": [], "diet_instructions": [], "followup": []}
        current_embedding = semantic_model.encode(chief_complaints.lower(), convert_to_tensor=True)
        
        # Track similarity scores for sorting medications
        medication_visits = []
        
        similar_visits_found = 0
        for visit in doctor_visits:
            visit_complaints = ""
            if "chief_complaints" in visit and isinstance(visit["chief_complaints"], dict):
                visit_complaints = visit["chief_complaints"].get("text", "").lower()
            elif "chief_complaints" in visit and isinstance(visit["chief_complaints"], str):
                visit_complaints = visit["chief_complaints"].lower()
            
            if not visit_complaints:
                continue
                
            visit_embedding = semantic_model.encode(visit_complaints, convert_to_tensor=True)
            similarity = util.cos_sim(current_embedding, visit_embedding).item()
            
            if similarity > 0.4:
                similar_visits_found += 1
                if "diagnosis" in visit and visit["diagnosis"]:
                    preferences["diagnosis"].extend(visit["diagnosis"])
                if "investigations" in visit and visit["investigations"]:
                    investigations = visit["investigations"].get("list", []) if isinstance(visit["investigations"], dict) else visit["investigations"]
                    preferences["investigations"].extend(investigations)
                if "medications" in visit and visit["medications"]:
                    # Store medications with similarity score for later sorting
                    medication_visits.append({"meds": visit["medications"], "similarity": similarity})
                if "diet_instructions" in visit and visit["diet_instructions"]:
                    content = visit["diet_instructions"].get("content", "") if isinstance(visit["diet_instructions"], dict) else visit["diet_instructions"]
                    preferences["diet_instructions"].append(content)
                if "followup" in visit and visit["followup"]:
                    preferences["followup"].append(visit["followup"])
        
        # Sort medications by similarity score and add to preferences
        if medication_visits:
            medication_visits.sort(key=lambda x: x["similarity"], reverse=True)
            for visit in medication_visits:
                preferences["medications"].extend(visit["meds"])
        
        logger.info(f"Found {similar_visits_found} semantically similar visits")
        return preferences
    except Exception as e:
        logger.error(f"Error fetching doctor preferences: {str(e)}")
        return {}

def extract_patient_details(past_visits):
    """Extract patient sections from the last visit only"""
    details = {
        "medical_history": "",
        "personal_history": "",
        "family_history": "",
        "current_medications": "",
        "allergies": ""
    }
    
    if not past_visits:
        logger.info("No past visits found for patient details extraction")
        return details

    # Use only the most recent visit
    latest_visit = past_visits[0]
    details["medical_history"] = latest_visit.get("medical_history", {}).get("content", "") if isinstance(latest_visit.get("medical_history"), dict) else latest_visit.get("medical_history", "")
    details["personal_history"] = latest_visit.get("personal_history", {}).get("content", "") if isinstance(latest_visit.get("personal_history"), dict) else latest_visit.get("personal_history", "")
    details["family_history"] = latest_visit.get("family_history", {}).get("content", "") if isinstance(latest_visit.get("family_history"), dict) else latest_visit.get("family_history", "")
    details["current_medications"] = latest_visit.get("current_medications", {}).get("content", "") if isinstance(latest_visit.get("current_medications"), dict) else latest_visit.get("current_medications", "")
    details["allergies"] = latest_visit.get("allergies", {}).get("content", "") if isinstance(latest_visit.get("allergies"), dict) else latest_visit.get("allergies", "")
    
    logger.info(f"Extracted patient details from latest visit: {details}")
    return details

# Modified process_llm_output for medications
def process_llm_output(text, field):
    """Process LLM output into structured format"""
    if not text:
        logger.warning(f"No text received for field {field}")
        if field == "medications":
            return EXAMPLE_CASES["medications"][-1]["output"]  # Fallback to example
        return None
    
    logger.info(f"Raw LLM output for {field}: {text}")
    
    # Clean up text by removing unwanted prefixes and extra quotes
    text = re.sub(r'^(INSTRUCTIONS|Output:)\s*', '', text.strip(), flags=re.IGNORECASE)
    
    # For medications, attempt multiple parsing strategies
    if field == "medications":
        # First attempt: Try to parse as JSON
        try:
            # Find and extract JSON array pattern
            json_match = re.search(r'\[.*?\]', text, re.DOTALL)
            if json_match:
                med_list = json.loads(json_match.group(0))
                # Validate structure
                valid_meds = []
                for med in med_list:
                    if isinstance(med, dict) and "name" in med and "instructions" in med:
                        valid_meds.append({
                            "name": med["name"].strip('"'),
                            "instructions": med["instructions"].strip('"')
                        })
                if valid_meds:
                    return valid_meds
        except (json.JSONDecodeError, AttributeError):
            pass  # Continue to next parsing strategy
            
        # Second attempt: Parse line by line format (Name: Instructions)
        try:
            med_list = []
            lines = text.split('\n')
            for line in lines:
                # Check for patterns like "Drug Name: Instructions" or "- Drug Name: Instructions"
                match = re.search(r'[-*]?\s*([^:]+):\s*(.+)', line)
                if match:
                    name = match.group(1).strip().strip('"')
                    instructions = match.group(2).strip().strip('"')
                    if name and instructions and "name" not in name.lower():
                        med_list.append({"name": name, "instructions": instructions})
            if med_list:
                return med_list
        except Exception:
            pass  # Continue to next parsing strategy
            
        # Third attempt: Try to extract medication names and instructions separately
        try:
            # Look for patterns like "1. Drug Name - Instructions" or "Drug Name - Instructions"
            med_list = []
            med_pattern = re.findall(r'(?:\d+\.\s*)?([^-\n]+)\s*[-:]\s*([^\n]+)', text)
            for name, instructions in med_pattern:
                name = name.strip().strip('"')
                instructions = instructions.strip().strip('"')
                if name and instructions:
                    med_list.append({"name": name, "instructions": instructions})
            if med_list:
                return med_list
        except Exception:
            pass
            
        # If all parsing attempts fail, return the default example
        logger.warning("Failed to parse medications, using default example")
        return EXAMPLE_CASES["medications"][-1]["output"]
    
    # For other fields, proceed with normal parsing
    json_pattern = r'\[.*\]|\{.*\}'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON for {field}: {text}")
    
    if field == "investigations":
        cleaned_text = ", ".join(line.strip().strip('"') for line in text.split(',') if line.strip())
        if "test1" in cleaned_text.lower() or not cleaned_text:
            return EXAMPLE_CASES["investigations"][-1]["output"]
        return cleaned_text
    if field == "diet_instructions":
        return text.strip().strip('"')
    
    return None


# Modified generate_prompt for medications
def generate_prompt(field, patient_data, doctor_prefs):
    """Generate LLM prompt for a specific field"""
    chief_complaints = patient_data.get("chief_complaints", {}).get("text", "")
    if not chief_complaints or "suffering from" in chief_complaints.lower():
        return ""  # Return empty prompt for invalid complaints

    base_context = f"""
Patient Information:
- Chief complaints: {chief_complaints}
- Medical history: {patient_data.get('medical_history', 'Not provided')}
- Personal history (habits): {patient_data.get('personal_history', 'Not provided')}
- Family history: {patient_data.get('family_history', 'Not provided')}
- Current medications: {patient_data.get('current_medications', 'Not provided')}
- Allergies: {patient_data.get('allergies', 'Not provided')}
- Vitals: {patient_data.get('vitals', 'Not provided')}
"""

    if doctor_prefs:
        base_context += "Doctor Preferences (use as reference only, prioritize current condition):\n"
        if doctor_prefs.get('diagnosis'):
            base_context += f"- Previous diagnoses: {', '.join([d.get('name', '') for d in doctor_prefs['diagnosis']][:3])}\n"
        if doctor_prefs.get('investigations'):
            base_context += f"- Previous investigations: {', '.join(doctor_prefs['investigations'][:5])}\n"
        if doctor_prefs.get('medications') and field == "medications":
            # Format medication examples more clearly
            medications = []
            for m in doctor_prefs['medications'][:5]:
                if isinstance(m, dict) and m.get('name') and m.get('instructions'):
                    medications.append(f"{m.get('name')}: {m.get('instructions')}")
            if medications:
                base_context += f"- Doctor's previous prescriptions for similar cases:\n"
                for i, med in enumerate(medications, 1):
                    base_context += f"  {i}. {med}\n"

        if doctor_prefs.get('diet_instructions'):
            base_context += f"- Previous diet instructions: {doctor_prefs['diet_instructions'][0][:50]}...\n"
        if doctor_prefs.get('followup'):
            base_context += f"- Previous followup: {doctor_prefs['followup'][0].get('text', '')[:50]}...\n"

    prompts = {
        "diagnosis": f"{base_context}\nProvide the most likely diagnosis based on current complaints and history.\nOutput format: [{{\"code\": \"ICD_CODE\", \"name\": \"DIAGNOSIS_NAME\"}}]\n{format_examples('diagnosis')}",
        "investigations": f"{base_context}\nProvide a concise, relevant comma-separated list of investigations for the current condition.\nOutput format: \"test1, test2, test3\"\n{format_examples('investigations')}",
        "medications": f"{base_context}\nProvide a list of medications with dosage instructions based on the patient's current complaints and history. Use doctor's previous prescriptions as a reference, but prioritize the current condition.\nOutput JSON format: [\n  {{\"name\": \"Medication Name\", \"instructions\": \"Dosage instructions\"}},\n  {{\"name\": \"Another Medication\", \"instructions\": \"Dosage instructions\"}}\n]\n{format_examples('medications')}",
        "followup": f"{base_context}\nProvide a followup plan based on current condition.\nOutput format: {{\"date\": \"YYYY-MM-DD\", \"text\": \"INSTRUCTIONS\"}}\n{format_examples('followup')}",
        "diet_instructions": f"{base_context}\nProvide concise diet instructions relevant to current condition as a single line.\nOutput format: \"INSTRUCTIONS\"\n{format_examples('diet_instructions')}"
    }
    
    return prompts.get(field, "")

def process_json_data(data):
    """Process input JSON and return completed data"""
    result = data.copy()
    patient_id = result.get("patientId")
    doctor_id = result.get("doctorId")
    
    # Validate request
    if not patient_id or not doctor_id:
        logger.error("Missing patientId or doctorId")
        return {"error": "Missing patientId or doctorId"}, 400

    # Step 1: Fetch patient history and fill patient sections
    past_visits = fetch_patient_history(patient_id)
    patient_details = extract_patient_details(past_visits)
    result.update(patient_details)

    # Step 2: Handle chief complaints
    chief_complaints = ""
    if "chief_complaints" in result:
        if isinstance(result["chief_complaints"], dict):
            chief_complaints = result["chief_complaints"].get("text", "")
        elif isinstance(result["chief_complaints"], str):
            chief_complaints = result["chief_complaints"]
            # Convert to expected format for consistency
            result["chief_complaints"] = {"text": chief_complaints}
    
    if not chief_complaints or "suffering from" in chief_complaints.lower():
        logger.info("Invalid chief complaints, returning patient details only")
        return result

    # Step 3: Fetch doctor preferences
    doctor_prefs = fetch_doctor_preferences(doctor_id, chief_complaints)

    # Step 4: Process fields with LLM
    doctor_fields = ["diagnosis", "investigations", "medications", "followup", "diet_instructions"]
    for field in doctor_fields:
        is_empty = (field not in result or 
                    (isinstance(result[field], list) and not result[field]) or 
                    (isinstance(result[field], str) and not result[field].strip()) or 
                    (isinstance(result[field], dict) and not any(result[field].values())))
        if is_empty:
            prompt = generate_prompt(field, result, doctor_prefs)
            if prompt:  # Only process if prompt is valid
                llm_output = get_llm_suggestion(prompt)
                if llm_output:
                    processed_output = process_llm_output(llm_output, field)
                    if processed_output is not None:
                        result[field] = processed_output
                        logger.info(f"Set {field} to: {processed_output}")
                else:
                    logger.warning(f"No LLM output received for {field}")

    # Step 5: Refine diagnosis with ICD-11 API
    if "diagnosis" in result and result["diagnosis"]:
        diagnosis_name = ""
        if isinstance(result["diagnosis"], list) and result["diagnosis"]:
            diagnosis_name = result["diagnosis"][0].get("name", "")
        elif isinstance(result["diagnosis"], dict):
            diagnosis_name = result["diagnosis"].get("name", "")
        
        if diagnosis_name:
            api_diagnosis = get_icd11_diagnosis(diagnosis_name)
            if api_diagnosis:
                result["diagnosis"] = [api_diagnosis]
                logger.info(f"Refined diagnosis with ICD-11 API: {api_diagnosis}")

    return result

@app.route('/api/suggest', methods=['POST'])
def suggest():
    """API endpoint to process input and provide suggestions"""
    try:
        data = request.json
        if not data or "patientId" not in data or "doctorId" not in data:
            return jsonify({"error": "Missing patientId or doctorId"}), 400
        
        completed_data = process_json_data(data)
        if isinstance(completed_data, tuple):  # Error case
            return jsonify(completed_data[0]), completed_data[1]
        return jsonify(completed_data)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)