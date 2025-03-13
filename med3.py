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

# Access to master collections
medications_collection = db["medications"]
investigations_collection = db["investigations"]
master_allergies_collection = db["masterallergies"]

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
        {"input": "Chief complaints: High blood pressure and chest pain.", "output": [{"code": "", "name": "ECG"}, {"code": "", "name": "Chest X-ray"}, {"code": "", "name": "Blood tests"}]},
        {"input": "Chief complaints: cold and ingestion", "output": [{"code": "", "name": "CBC"}, {"code": "", "name": "Chest X-ray"}, {"code": "", "name": "Allergy testing"}]},
        {"input": "Chief complaints: rash and itching", "output": [{"code": "", "name": "CBC"}, {"code": "", "name": "Skin allergy test"}, {"code": "", "name": "IgE levels"}]},
        {"input": "Chief complaints: Ae JCB hit the patient on ribs area in the previous xrays primary xray shows there are no fractures and cracks in the ribs but the patient is feeling severe pain.while moving his hand ,inhaling,exhaling he is not able to sleep on any side.", "output": [{"code": "", "name": "Chest CT"}, {"code": "", "name": "Repeat X-ray"}, {"code": "", "name": "Ultrasound of chest"}]},
    ],
    "medications": [
        {"input": "Chief complaints: High blood pressure and chest pain. Medical history: Hypertension.", "output": [
            {"code": "", "name": "Amlodipine"},
            {"code": "", "name": "Nitroglycerin"}
        ]},
        {"input": "Chief complaints: Sinussitis and heavy cold.", "output": [
            {"code": "", "name": "Amoxicillin"}
        ]},
        {"input": "Chief complaints: Ae JCB hit the patient on ribs area in the previous xrays primary xray shows there are no fractures and cracks in the ribs but the patient is feeling severe pain.while moving his hand ,inhaling,exhaling he is not able to sleep on any side.", "output": [
            {"code": "", "name": "Ibuprofen"},
            {"code": "", "name": "Acetaminophen"}
        ]},
    ],
    "allergies": [
        {"input": "Chief complaints: rash and itching after eating seafood.", "output": [
            {"code": "", "name": "Seafood"}
        ]},
        {"input": "Chief complaints: sneezing and itchy eyes during spring.", "output": [
            {"code": "", "name": "Pollen"},
            {"code": "", "name": "Dust"}
        ]}
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

def validate_chief_complaints(chief_complaints):
    """Validate if the chief complaints are specific and medically relevant using LLM"""
    if not chief_complaints or not isinstance(chief_complaints, str) or not chief_complaints.strip():
        logger.info(f"Chief complaint is empty or invalid: {chief_complaints}")
        return False

    prompt = f"""
You are a medical AI assistant specialized in validating chief complaints. Your task is to determine if a patient's chief complaint is valid for medical documentation.

A chief complaint is VALID if it:
- Describes specific symptoms (e.g., fever, pain, cough)
- Mentions medical conditions (e.g., hypertension, diabetes)
- Describes injuries or trauma (e.g., fall, accident)
- Includes phrases like "the patient has" followed by symptoms
- Can be brief but must contain specific medical information

A chief complaint is INVALID if it:
- Is vague without specific symptoms (e.g., "feeling unwell", "not good")
- Contains no medical information
- Is a greeting or procedural text (e.g., "need appointment", "hello")
- Contains only administrative information

You must respond with ONLY "true" or "false".

Examples of VALID chief complaints:
- "The patient has high fever" → true
- "High fever" → true
- "The patient has chest pain and difficulty breathing" → true
- "Headache for 3 days" → true
- "The patient has cough and cold" → true
- "The patient has diabetes" → true
- "Abdominal pain since yesterday" → true
- "Patient fell and injured knee" → true

Examples of INVALID chief complaints:
- "The patient is here for checkup" → false
- "Need medicine" → false
- "Not feeling well" → false
- "Hello doctor" → false
- "Need appointment" → false
- "The patient is here" → false
- "Review" → false

Chief complaint to validate: "{chief_complaints}"

Valid (true) or Invalid (false):
"""
    llm_output = get_llm_suggestion(prompt)
    if llm_output:
        logger.info(f"LLM validation result for '{chief_complaints}': {llm_output}")
        # Extract just the true/false from the response, ignoring any explanations
        result = "true" in llm_output.strip().lower()
        if not result:
            logger.info(f"Chief complaint '{chief_complaints}' deemed invalid by LLM")
        return result
    else:
        logger.info(f"LLM failed for '{chief_complaints}', returning False")
        return False

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

def find_in_collection(name, collection, similarity_threshold=0.75):
    """Find an item in a collection using exact match or semantic similarity"""
    try:
        # First try direct match in name field
        exact_match = collection.find_one({"name": {"$regex": f"^{re.escape(name)}$", "$options": "i"}})
        if exact_match:
            return {"code": str(exact_match["_id"]), "name": exact_match["name"]}
        
        # Also try matching on generic_name if it exists
        generic_match = collection.find_one({"generic_name": {"$regex": f"^{re.escape(name)}$", "$options": "i"}})
        if generic_match:
            return {"code": str(generic_match["_id"]), "name": generic_match["name"]}
        
        # If no exact match, use semantic search
        name_embedding = semantic_model.encode(name.lower(), convert_to_tensor=True)
        
        # Get all items from collection (optimally this would be paginated or limited)
        items = list(collection.find({}).limit(500))
        
        best_match = None
        highest_similarity = similarity_threshold
        
        for item in items:
            item_name = item.get("name", "")
            if not item_name:
                continue
                
            item_embedding = semantic_model.encode(item_name.lower(), convert_to_tensor=True)
            similarity = util.cos_sim(name_embedding, item_embedding).item()
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = item
            
            # Also check generic_name if available
            if "generic_name" in item and item["generic_name"]:
                generic_embedding = semantic_model.encode(item["generic_name"].lower(), convert_to_tensor=True)
                generic_similarity = util.cos_sim(name_embedding, generic_embedding).item()
                
                if generic_similarity > highest_similarity:
                    highest_similarity = generic_similarity
                    best_match = item
        
        if best_match:
            logger.info(f"Found semantic match for '{name}' with similarity {highest_similarity}: {best_match['name']}")
            return {"code": str(best_match["_id"]), "name": best_match["name"]}
        
        # No match found
        logger.info(f"No match found for '{name}' in collection")
        return {"code": "", "name": name}
        
    except Exception as e:
        logger.error(f"Error searching collection for '{name}': {str(e)}")
        return {"code": "", "name": name}

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
        
        preferences = {"diagnosis": [], "investigations": [], "medications": [], "diet_instructions": [], "followup": [], "allergies": []}
        current_embedding = semantic_model.encode(chief_complaints.lower(), convert_to_tensor=True)
        
        best_medication_match = None
        highest_similarity = 0.4  # Minimum threshold
        
        for visit in doctor_visits:
            visit_complaints = ""
            if "chief_complaints" in visit and isinstance(visit["chief_complaints"], dict):
                visit_complaints = visit["chief_complaints"].get("text", "").lower()
            elif "chief_complaints" in visit and isinstance(visit["chief_complaints"], str):
                visit_complaints = visit["chief_complaints"].lower()
            elif "chief_complaints" in visit and isinstance(visit["chief_complaints"], list) and visit["chief_complaints"]:
                visit_complaints = visit["chief_complaints"][0].get("text", "").lower()
            
            if not visit_complaints:
                continue
                
            visit_embedding = semantic_model.encode(visit_complaints, convert_to_tensor=True)
            similarity = util.cos_sim(current_embedding, visit_embedding).item()
            
            if similarity > highest_similarity and "medications" in visit and visit["medications"]:
                highest_similarity = similarity
                best_medication_match = visit["medications"]
                preferences["medications"] = visit["medications"]
            
            if similarity > 0.4:
                if "diagnosis" in visit and visit["diagnosis"]:
                    preferences["diagnosis"].extend(visit["diagnosis"])
                if "investigations" in visit and visit["investigations"]:
                    investigations = visit["investigations"].get("list", []) if isinstance(visit["investigations"], dict) else visit["investigations"]
                    preferences["investigations"].extend(investigations)
                if "diet_instructions" in visit and visit["diet_instructions"]:
                    content = visit["diet_instructions"].get("content", "") if isinstance(visit["diet_instructions"], dict) else visit["diet_instructions"]
                    preferences["diet_instructions"].append(content)
                if "followup" in visit and visit["followup"]:
                    preferences["followup"].append(visit["followup"])
                if "allergies" in visit and visit["allergies"]:
                    allergies = visit["allergies"].get("list", []) if isinstance(visit["allergies"], dict) else visit["allergies"]
                    preferences["allergies"].extend(allergies)
        
        logger.info(f"Best medication similarity score: {highest_similarity}")
        return preferences
    except Exception as e:
        logger.error(f"Error fetching doctor preferences: {str(e)}")
        return {}

def extract_patient_details(past_visits):
    """Extract patient sections from the last visit only, wrapping text in array of objects"""
    details = {
        "medical_history": [{"text": ""}],
        "personal_history": [{"text": ""}],
        "family_history": [{"text": ""}],
        "current_medications": [{"text": ""}],
        "allergies": [{"code": "", "name": ""}],
    }
    
    if not past_visits:
        logger.info("No past visits found for patient details extraction")
        return details

    latest_visit = past_visits[0]
    details["medical_history"] = [{"text": latest_visit.get("medical_history", {}).get("content", "") if isinstance(latest_visit.get("medical_history"), dict) else latest_visit.get("medical_history", "")}]
    details["personal_history"] = [{"text": latest_visit.get("personal_history", {}).get("content", "") if isinstance(latest_visit.get("personal_history"), dict) else latest_visit.get("personal_history", "")}]
    details["family_history"] = [{"text": latest_visit.get("family_history", {}).get("content", "") if isinstance(latest_visit.get("family_history"), dict) else latest_visit.get("family_history", "")}]
    details["current_medications"] = [{"text": latest_visit.get("current_medications", {}).get("content", "") if isinstance(latest_visit.get("current_medications"), dict) else latest_visit.get("current_medications", "")}]
    
    # Convert allergies to new format
    allergies_data = []
    raw_allergies = latest_visit.get("allergies", [])
    
    if isinstance(raw_allergies, dict) and "content" in raw_allergies:
        allergies_text = raw_allergies["content"]
        if allergies_text:
            for allergy in allergies_text.split(","):
                allergy = allergy.strip()
                if allergy:
                    matched_allergy = find_in_collection(allergy, master_allergies_collection)
                    allergies_data.append(matched_allergy)
    elif isinstance(raw_allergies, str):
        for allergy in raw_allergies.split(","):
            allergy = allergy.strip()
            if allergy:
                matched_allergy = find_in_collection(allergy, master_allergies_collection)
                allergies_data.append(matched_allergy)
    elif isinstance(raw_allergies, list):
        for allergy_item in raw_allergies:
            if isinstance(allergy_item, dict) and "name" in allergy_item:
                allergies_data.append(allergy_item)
            elif isinstance(allergy_item, str):
                matched_allergy = find_in_collection(allergy_item, master_allergies_collection)
                allergies_data.append(matched_allergy)
    
    if allergies_data:
        details["allergies"] = allergies_data
    
    logger.info(f"Extracted patient details from latest visit")
    return details

def process_llm_output(text, field):
    """Process LLM output into structured format"""
    if not text:
        logger.warning(f"No text received for field {field}")
        if field == "medications":
            return EXAMPLE_CASES["medications"][-1]["output"]
        return None
    
    logger.info(f"Raw LLM output for {field}: {text}")
    
    text = re.sub(r'^(INSTRUCTIONS|Output:)\s*', '', text.strip(), flags=re.IGNORECASE)
    text = text.strip('"')
    text = re.sub(r'\n+', '', text).strip()
    
    if field == "medications":
        try:
            # Try to extract JSON array
            json_pattern = r'\[.*\]'
            match = re.search(json_pattern, text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                
                # Process each medication to find matches in collection
                processed_medications = []
                for item in parsed:
                    if isinstance(item, dict) and "name" in item:
                        med_name = str(item["name"]).strip().strip('"')
                        matched_med = find_in_collection(med_name, medications_collection)
                        processed_medications.append(matched_med)
                
                return processed_medications if processed_medications else [{"code": "", "name": "Paracetamol"}]
            
            # Fallback parsing for non-JSON output
            meds = []
            for med_name in re.findall(r'"([^"]+)"', text):
                if med_name and not re.match(r'instructions|name|code', med_name, re.IGNORECASE):
                    matched_med = find_in_collection(med_name, medications_collection)
                    meds.append(matched_med)
            
            # If still no medications found, extract first words that might be medication names
            if not meds:
                potential_meds = re.findall(r'(?:^|,\s*)([A-Za-z]+)', text)
                for med_name in potential_meds[:3]:  # Take up to 3 potential medications
                    matched_med = find_in_collection(med_name, medications_collection)
                    meds.append(matched_med)
            
            return meds if meds else [{"code": "", "name": "Paracetamol"}]
        
        except Exception as e:
            logger.warning(f"Error processing medications output: {e}")
            return [{"code": "", "name": "Paracetamol"}]
    
    if field == "investigations":
        try:
            # Try to extract JSON array
            json_pattern = r'\[.*\]'
            match = re.search(json_pattern, text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                
                # Process each investigation to find matches in collection
                processed_investigations = []
                for item in parsed:
                    if isinstance(item, dict) and "name" in item:
                        inv_name = str(item["name"]).strip().strip('"')
                        matched_inv = find_in_collection(inv_name, investigations_collection)
                        processed_investigations.append(matched_inv)
                    elif isinstance(item, str):
                        matched_inv = find_in_collection(item.strip(), investigations_collection)
                        processed_investigations.append(matched_inv)
                
                return processed_investigations if processed_investigations else [{"code": "", "name": "CBC"}]
            
            # Fallback for non-JSON output - split by commas and process each investigation
            investigations = []
            for inv_name in re.split(r',\s*', text):
                inv_name = inv_name.strip().strip('"')
                if inv_name:
                    matched_inv = find_in_collection(inv_name, investigations_collection)
                    investigations.append(matched_inv)
            
            return investigations if investigations else [{"code": "", "name": "CBC"}]
        
        except Exception as e:
            logger.warning(f"Error processing investigations output: {e}")
            return [{"code": "", "name": "CBC"}]
    
    if field == "allergies":
        try:
            # Try to extract JSON array
            json_pattern = r'\[.*\]'
            match = re.search(json_pattern, text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                
                # Process each allergy to find matches in collection
                processed_allergies = []
                for item in parsed:
                    if isinstance(item, dict) and "name" in item:
                        allergy_name = str(item["name"]).strip().strip('"')
                        matched_allergy = find_in_collection(allergy_name, master_allergies_collection)
                        processed_allergies.append(matched_allergy)
                    elif isinstance(item, str):
                        matched_allergy = find_in_collection(item.strip(), master_allergies_collection)
                        processed_allergies.append(matched_allergy)
                
                return processed_allergies if processed_allergies else [{"code": "", "name": "Dust"}]
            
            # Fallback for non-JSON output - split by commas and process each allergy
            allergies = []
            for allergy_name in re.split(r',\s*', text):
                allergy_name = allergy_name.strip().strip('"')
                if allergy_name:
                    matched_allergy = find_in_collection(allergy_name, master_allergies_collection)
                    allergies.append(matched_allergy)
            
            return allergies if allergies else [{"code": "", "name": "Dust"}]
        
        except Exception as e:
            logger.warning(f"Error processing allergies output: {e}")
            return [{"code": "", "name": "Dust"}]
    
    if field == "diet_instructions":
        return [{"text": text.strip().strip('"')}]
    
    if field == "followup":
        try:
            # Try to extract JSON object
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            
            # Fallback - create a default followup
            return {"date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"), "text": text.strip().strip('"')}
        except Exception as e:
            logger.warning(f"Error processing followup output: {e}")
            return {"date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"), "text": "Follow up in 1 week."}
    
    return None

def generate_prompt(field, patient_data, doctor_prefs):
    """Generate LLM prompt for a specific field"""
    chief_complaints_list = patient_data.get("chief_complaints", [{"text": ""}])
    chief_complaints = chief_complaints_list[0].get("text", "") if chief_complaints_list else ""

    if not chief_complaints or not validate_chief_complaints(chief_complaints):
        return ""

    base_context = f"""
Patient Information:
- Chief complaints: {chief_complaints}
"""

    if doctor_prefs:
        base_context += "Doctor Preferences (use as reference only, prioritize current condition):\n"
        if doctor_prefs.get('diagnosis'):
            base_context += f"- Previous diagnoses: {', '.join([d.get('name', '') for d in doctor_prefs['diagnosis']][:3])}\n"
        if doctor_prefs.get('investigations'):
            base_context += f"- Previous investigations: {', '.join([i.get('name', '') if isinstance(i, dict) else i for i in doctor_prefs['investigations']])[:5]}\n"
        if field == "medications" and doctor_prefs.get('medications'):
            medications = doctor_prefs['medications']
            if medications:
                med_str = ", ".join([m.get('name', '') if isinstance(m, dict) else m for m in medications[:3]])
                base_context += f"- Previous medications for similar cases: {med_str}\n"
                base_context += "Refine these medications based on the current condition if needed.\n"
        if doctor_prefs.get('diet_instructions'):
            base_context += f"- Previous diet instructions: {doctor_prefs['diet_instructions'][0][:50]}...\n"
        if doctor_prefs.get('followup'):
            base_context += f"- Previous followup: {doctor_prefs['followup'][0].get('text', '')[:50]}...\n"
        if doctor_prefs.get('allergies'):
            base_context += f"- Previous allergies: {', '.join([a.get('name', '') if isinstance(a, dict) else a for a in doctor_prefs['allergies']])[:5]}\n"

    prompts = {
        "diagnosis": f"{base_context}\nProvide the most likely diagnosis based on current complaints and history.\nOutput format: [{{\"code\": \"ICD_CODE\", \"name\": \"DIAGNOSIS_NAME\"}}]\n{format_examples('diagnosis')}",
        "investigations": f"{base_context}\nProvide a list of investigations needed for the current condition.\nOutput format: [{{\"code\": \"\", \"name\": \"INVESTIGATION_NAME\"}}]\n{format_examples('investigations')}",
        "medications": f"{base_context}\nProvide a list of medications based on the patient's current complaints and history. If previous medications are provided, refine them for the current condition; otherwise, suggest new ones.\nOutput format: [{{\"code\": \"\", \"name\": \"MEDICATION_NAME\"}}]\n{format_examples('medications')}",
        "followup": f"{base_context}\nProvide a followup plan based on current condition.\nOutput format: {{\"date\": \"YYYY-MM-DD\", \"text\": \"INSTRUCTIONS\"}}\n{format_examples('followup')}",
        "diet_instructions": f"{base_context}\nProvide concise diet instructions relevant to current condition as a single line.\nOutput format: \"INSTRUCTIONS\"\n{format_examples('diet_instructions')}",
        "allergies": f"{base_context}\nProvide potential allergies the patient might have based on their complaints and history.\nOutput format: [{{\"code\": \"\", \"name\": \"ALLERGY_NAME\"}}]\n{format_examples('allergies')}"
    }
    
    return prompts.get(field, "")

def process_json_data(data):
    """Process input JSON and return completed data with text fields as arrays of objects"""
    result = data.copy()
    patient_id = result.get("patientId")
    doctor_id = result.get("doctorId")
    
    if not patient_id or not doctor_id:
        logger.error("Missing patientId or doctorId")
        return {"error": "Missing patientId or doctorId"}, 400

    # Step 1: Fetch patient history and fill patient sections
    past_visits = fetch_patient_history(patient_id)
    patient_details = extract_patient_details(past_visits)
    result.update(patient_details)

    # Step 2: Handle chief complaints and validate
    chief_complaints = ""
    if "chief_complaints" in result:
        if isinstance(result["chief_complaints"], dict):
            chief_complaints = result["chief_complaints"].get("text", "")
        elif isinstance(result["chief_complaints"], str):
            chief_complaints = result["chief_complaints"]
        elif isinstance(result["chief_complaints"], list) and result["chief_complaints"]:
            chief_complaints = result["chief_complaints"][0].get("text", "")
        result["chief_complaints"] = [{"text": chief_complaints}]
    
    # Validate chief complaints
    if not validate_chief_complaints(chief_complaints):
        logger.info(f"Invalid chief complaints: '{chief_complaints}', returning patient details only")
        return result

    # Step 3: Fetch doctor preferences
    doctor_prefs = fetch_doctor_preferences(doctor_id, chief_complaints)

    # Step 4: Process fields with LLM
    doctor_fields = ["diagnosis", "investigations", "medications", "followup", "diet_instructions", "allergies"]
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
                        if field == "diet_instructions" and isinstance(processed_output, str):
                            result[field] = [{"text": processed_output}]
                        else:
                            result[field] = processed_output
                        logger.info(f"Set {field} to: {result[field]}")
                else:
                    logger.warning(f"No LLM output received for {field}")

    # Step 5: Process investigations if they exist in text format
    if "investigations" in result:
        if isinstance(result["investigations"], list):
            # Check if it's in the old format (list of dicts with text field)
            if result["investigations"] and isinstance(result["investigations"][0], dict) and "text" in result["investigations"][0]:
                investigation_names = []
                for item in result["investigations"]:
                    if isinstance(item, dict) and "text" in item:
                        text = item["text"]
                        # Split by commas if multiple investigations are in one text field
                        for inv_name in re.split(r',\s*', text):
                            inv_name = inv_name.strip()
                            if inv_name:
                                investigation_names.append(inv_name)
                
                processed_investigations = []
                for name in investigation_names:
                    matched_inv = find_in_collection(name, investigations_collection)
                    processed_investigations.append(matched_inv)
                
                result["investigations"] = processed_investigations if processed_investigations else [{"code": "", "name": "CBC"}]
        elif isinstance(result["investigations"], str):
            investigation_names = re.split(r',\s*', result["investigations"])
            processed_investigations = []
            for name in investigation_names:
                if name.strip():
                    matched_inv = find_in_collection(name.strip(), investigations_collection)
                    processed_investigations.append(matched_inv)
            
            result["investigations"] = processed_investigations if processed_investigations else [{"code": "", "name": "CBC"}]

    # Step 6: Process medications if they exist in text format
    if "medications" in result:
        if isinstance(result["medications"], list):
            # Check if it's in the old format (list of dicts with text field)
            if result["medications"] and isinstance(result["medications"][0], dict) and "text" in result["medications"][0]:
                medication_names = []
                for item in result["medications"]:
                    if isinstance(item, dict) and "text" in item:
                        text = item["text"]
                        # Split by commas if multiple medications are in one text field
                        for med_name in re.split(r',\s*', text):
                            med_name = med_name.strip()
                            if med_name:
                                medication_names.append(med_name)
                
                processed_medications = []
                for name in medication_names:
                    matched_med = find_in_collection(name, medications_collection)
                    processed_medications.append(matched_med)
                
                result["medications"] = processed_medications if processed_medications else [{"code": "", "name": "Paracetamol"}]
        elif isinstance(result["medications"], str):
            medication_names = re.split(r',\s*', result["medications"])
            processed_medications = []
            for name in medication_names:
                if name.strip():
                    matched_med = find_in_collection(name.strip(), medications_collection)
                    processed_medications.append(matched_med)
            
            result["medications"] = processed_medications if processed_medications else [{"code": "", "name": "Paracetamol"}]

    # Step 7: Process allergies if they exist in text format
    if "allergies" in result:
        if isinstance(result["allergies"], list):
            # Check if it's already in the new format or needs conversion
            if result["allergies"] and isinstance(result["allergies"][0], dict) and "text" in result["allergies"][0]:
                allergy_names = []
                for item in result["allergies"]:
                    if isinstance(item, dict) and "text" in item:
                        text = item["text"]
                        # Split by commas if multiple allergies are in one text field
                        for allergy_name in re.split(r',\s*', text):
                            allergy_name = allergy_name.strip()
                            if allergy_name:
                                allergy_names.append(allergy_name)
                
                processed_allergies = []
                for name in allergy_names:
                    matched_allergy = find_in_collection(name, master_allergies_collection)
                    processed_allergies.append(matched_allergy)
                
                result["allergies"] = processed_allergies if processed_allergies else [{"code": "", "name": "Dust"}]
        elif isinstance(result["allergies"], str):
            allergy_names = re.split(r',\s*', result["allergies"])
            processed_allergies = []
            for name in allergy_names:
                if name.strip():
                    matched_allergy = find_in_collection(name.strip(), master_allergies_collection)
                    processed_allergies.append(matched_allergy)
            
            result["allergies"] = processed_allergies if processed_allergies else [{"code": "", "name": "Dust"}]

    # Step 8: Refine diagnosis with ICD-11 API if needed
    if "diagnosis" in result and result["diagnosis"]:
        diagnosis_name = ""
        if isinstance(result["diagnosis"], list) and result["diagnosis"]:
            diagnosis_name = result["diagnosis"][0].get("name", "")
        elif isinstance(result["diagnosis"], dict):
            diagnosis_name = result["diagnosis"].get("name", "")
        
        if diagnosis_name and not result["diagnosis"][0].get("code"):
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