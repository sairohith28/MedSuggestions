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
import numpy as np
import torch
from bson import ObjectId
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["chatbot_dev"]
patient_visits_collection = db["patient-visits"]
allergies_collection = db["masterallergies"]
medications_collection = db["medications"]
drug_interactions_collection = db["drug-interactions"]
investigations_collection = db["investigations"]

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
        {"input": "Chief complaints: Ae JCB hit the patient on ribs area...", "output": [{"code": "S29.9", "name": "Unspecified injury of thorax"}]},
    ],
    "investigations": [
        {"input": "Chief complaints: High blood pressure and chest pain.", "output": [{"code": "", "name": "ECG"}, {"code": "", "name": "Chest X-ray"}]},
        {"input": "Chief complaints: cold and ingestion", "output": [{"code": "", "name": "CBC"}, {"code": "", "name": "Chest X-ray"}]},
        {"input": "Chief complaints: rash and itching", "output": [{"code": "", "name": "CBC"}, {"code": "", "name": "Skin allergy test"}]},
        {"input": "Chief complaints: Ae JCB hit the patient on ribs area...", "output": [{"code": "", "name": "Chest CT"}, {"code": "", "name": "Repeat X-ray"}]},
    ],
    "medications": [
        {"input": "Chief complaints: High blood pressure and chest pain...", "output": [
            {"code": "", "name": "Amlodipine"}, {"code": "", "name": "Nitroglycerin"}
        ]},
        {"input": "Chief complaints: Sinussitis and heavy cold.", "output": [{"code": "", "name": "Amoxicillin"}]},
        {"input": "Chief complaints: Ae JCB hit the patient on ribs area...", "output": [
            {"code": "", "name": "Ibuprofen"}, {"code": "", "name": "Acetaminophen"}
        ]},
    ],
    "followup": [
        {"input": "Chief complaints: High blood pressure and chest pain.", "output": {"date": "2025-03-27", "text": "Review in 1 week with ECG results."}},
        {"input": "Chief complaints: cold and ingestion", "output": {"date": "2025-03-25", "text": "Review in 5 days."}},
        {"input": "Chief complaints: Ae JCB hit the patient on ribs area...", "output": {"date": "2025-03-27", "text": "Review in 1 week with imaging results."}},
    ],
    "diet_instructions": [
        {"input": "Chief complaints: High blood pressure and chest pain.", "output": "Reduce salt intake, avoid fatty foods, increase potassium-rich foods"},
        {"input": "Chief complaints: cold and ingestion", "output": "Stay hydrated, drink plenty of fluids, avoid spicy foods"},
        {"input": "Chief complaints: Ae JCB hit the patient on ribs area...", "output": "Eat soft foods, avoid heavy lifting, maintain light diet"}
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

Chief complaint to validate: "{chief_complaints}"

Valid (true) or Invalid (false):
"""
    llm_output = get_llm_suggestion(prompt)
    if llm_output:
        logger.info(f"LLM validation result for '{chief_complaints}': '{llm_output}'")
        cleaned_output = llm_output.strip().lower()
        is_valid = "true" in cleaned_output or "valid" in cleaned_output
        if not is_valid:
            logger.info(f"Chief complaint '{chief_complaints}' deemed invalid by LLM: '{llm_output}'")
        return is_valid
    logger.warning(f"LLM returned no output for '{chief_complaints}', assuming invalid")
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

def search_collection(collection, search_term, field="name", use_embeddings=False, branchId=None, organizationId=None):
    """Search a collection using regex or embeddings with branchId and organizationId filters"""
    try:
        if not search_term or not isinstance(search_term, str):
            return {"code": "", "name": search_term or ""}

        # Enforce branchId and organizationId for medications and investigations
        if collection.name in ["medications", "investigations"]:
            if not branchId or not organizationId:
                logger.warning(f"Missing branchId or organizationId for {collection.name} search: {search_term}")
                return {"code": "", "name": search_term}

        query = {}
        if collection.name in ["medications", "investigations"]:
            query["branchId"] = branchId
            query["organizationId"] = ObjectId(organizationId)

        if use_embeddings and collection.name == "medications":
            query_embedding = semantic_model.encode(search_term.lower(), convert_to_tensor=True).cpu().numpy().tolist()
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "default",
                        "path": "name_generic_embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 100,
                        "limit": 5
                    }
                },
                {
                    "$match": {
                        "branchId": branchId,
                        "organizationId": ObjectId(organizationId)
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "name": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            results = list(collection.aggregate(pipeline))
            if results:
                top_match = results[0]
                if top_match.get("score") >= 0.7:
                    return {"code": str(top_match["_id"]), "name": top_match["name"]}
            return {"code": "", "name": search_term}

        regex_pattern = re.compile(re.escape(search_term), re.IGNORECASE)
        query["$or"] = [
            {field: {"$regex": regex_pattern}},
            {"generic_name": {"$regex": regex_pattern}} if collection.name == "medications" else {}
        ]
        if not query["$or"][-1]:
            query["$or"].pop()

        result = collection.find_one(query)
        if result:
            return {"code": str(result["_id"]), "name": result["name"]}
        return {"code": "", "name": search_term}
    except Exception as e:
        logger.error(f"Error searching collection {collection.name}: {str(e)}")
        return {"code": "", "name": search_term}

def check_drug_interactions(medications):
    """Check for drug-to-drug interactions using vector search, then filter for exact matches"""
    interactions = []
    
    if len(medications) <= 1:
        return interactions
    
    med_names = [med["name"].lower() for med in medications if med.get("name")]
    med_names_set = set(med_names)
    
    for i, med_a in enumerate(med_names):
        for j, med_b in enumerate(med_names[i+1:], start=i+1):
            interaction_query = f"{med_a} {med_b}"
            query_embedding = semantic_model.encode(interaction_query, convert_to_tensor=True).cpu().numpy().tolist()
            
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "default",
                        "path": "interaction_embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 50,
                        "limit": 3
                    }
                },
                {
                    "$project": {
                        "Drug_A": 1,
                        "Drug_B": 1,
                        "Level": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = list(drug_interactions_collection.aggregate(pipeline))
            
            for result in results:
                if result.get("score") >= 0.75:  # Removed Level check
                    drug_a = result["Drug_A"].lower()
                    drug_b = result["Drug_B"].lower()
                    if drug_a in med_names_set and drug_b in med_names_set:
                        level = result.get("Level", "Unknown")  # Use Unknown if Level is missing
                        interactions.append({
                            "drug_a": med_a,
                            "drug_b": med_b,
                            "level": level,
                            "suggestion": f"{med_a}-{med_b} combination has a {level} interaction"
                        })
                        break
    
    return interactions

def fetch_patient_history(patient_id):
    """Fetch last 10 consultations for patient"""
    try:
        visits = list(patient_visits_collection.find({"patientId": ObjectId(patient_id)}).sort("createdAt", -1).limit(10))
        return visits
    except Exception as e:
        logger.error(f"Error fetching patient history: {str(e)}")
        return []

def fetch_doctor_preferences(doctor_id, chief_complaints):
    """Fetch doctor preferences using semantic similarity - use only for reference"""
    if not chief_complaints or not isinstance(chief_complaints, str) or not chief_complaints.strip():
        return {}
    
    try:
        visits = patient_visits_collection.find({"doctorId": ObjectId(doctor_id)}).sort("createdAt", -1).limit(50)
        doctor_visits = list(visits)
        
        preferences = {"diagnosis": [], "investigations": [], "medications": [], "diet_instructions": [], "followup": [], "scores":[], "conditions":[]}
        current_embedding = semantic_model.encode(chief_complaints.lower(), convert_to_tensor=True)
        
        for visit in doctor_visits:
            visit_complaints = visit.get("chief_complaints", {}).get("content", "").lower() if isinstance(visit.get("chief_complaints"), dict) else visit.get("chief_complaints", "").lower()
            if not visit_complaints:
                continue
                
            visit_embedding = semantic_model.encode(visit_complaints, convert_to_tensor=True)
            similarity = util.cos_sim(current_embedding, visit_embedding).item()
            
            # Only consider highly similar conditions to avoid inappropriate medication suggestions
            if similarity > 0.6:  # Increased threshold to ensure better relevance
                # Store the condition along with every medication to provide context 
                if "medications" in visit and visit["medications"]:
                    medications = [item["name"] for item in visit["medications"]] if isinstance(visit["medications"], list) else visit["medications"]
                    preferences["medications"].extend(medications)
                    preferences["conditions"].extend([visit_complaints] * len(medications))
                    preferences['scores'].extend([similarity] * len(medications))
                
                # Still collect these for reference
                if "diagnosis" in visit and visit["diagnosis"]:
                    preferences["diagnosis"].extend(visit["diagnosis"])
                if "investigations" in visit and visit["investigations"]:
                    investigations = [item["name"] for item in visit["investigations"]] if isinstance(visit["investigations"], list) else visit["investigations"]
                    preferences["investigations"].extend(investigations)
                if "diet_instructions" in visit and visit["diet_instructions"]:
                    content = visit["diet_instructions"][0]["text"] if isinstance(visit["diet_instructions"], list) and visit["diet_instructions"] else visit["diet_instructions"]
                    preferences["diet_instructions"].append(content)
                if "followup" in visit and visit["followup"]:
                    preferences["followup"].append(visit["followup"])
        
        return preferences
    except Exception as e:
        logger.error(f"Error fetching doctor preferences: {str(e)}")
        return {}

def extract_patient_details(past_visits, input_data=None):
    """Extract patient sections from the last visit only"""
    # Return empty object as we're removing all these fields
    return {}

def process_llm_output(text, field):
    """Process LLM output into structured format"""
    if not text:
        logger.warning(f"No text received for field {field}")
        if field == "medications":
            return [{"code": "", "name": "Paracetamol"}]
        return None
    
    text = re.sub(r'^(INSTRUCTIONS|Output:)\s*', '', text.strip(), flags=re.IGNORECASE)
    text = text.strip('"').lstrip(':').strip()  # Ensure colon is removed at the start
    
    if field in ["medications"]:
        json_pattern = r'\[.*\]'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                items = [{"code": "", "name": item["name"]} for item in parsed if isinstance(item, dict) and "name" in item]
                seen = set()
                items = [item for item in items if not (item["name"] in seen or seen.add(item["name"]))]
                return items
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON for {field}: {text}")
        
        items = []
        for part in text.split("},"):
            name_match = re.search(r'"name":\s*"([^"]+)"', part)
            if name_match:
                items.append({"code": "", "name": name_match.group(1)})
        seen = set()
        items = [item for item in items if not (item["name"] in seen or seen.add(item["name"]))]
        return items if items else [{"code": "", "name": "Paracetamol" if field == "medications" else ""}]
    
    if field == "investigations":
        json_pattern = r'\[.*\]'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [{"code": "", "name": item["name"]} if isinstance(item, dict) else {"code": "", "name": item} for item in parsed]
                return [{"code": "", "name": parsed}]
            except json.JSONDecodeError:
                pass
        cleaned_text = [line.strip().strip('"') for line in text.split(',') if line.strip()]
        return [{"code": "", "name": item} for item in cleaned_text] if cleaned_text else [{"code": "", "name": "CBC"}]
    
    json_pattern = r'\[.*\]|\{.*\}'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if field == "diet_instructions" and isinstance(parsed, str):
                return [{"text": parsed.lstrip(':').strip()}]  # Ensure colon is removed
            if field == "followup" and isinstance(parsed, dict):
                current_date = datetime(2025, 3, 20)
                followup_date = datetime.strptime(parsed["date"], "%Y-%m-%d")
                if followup_date <= current_date:
                    days_to_add = 7 if "week" in parsed["text"].lower() else 5
                    parsed["date"] = (current_date + timedelta(days=days_to_add)).strftime("%Y-%m-%d")
                return parsed
            return parsed
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON for {field}: {text}")
    
    if field == "diet_instructions":
        return [{"text": text.lstrip(':').strip()}]  # Ensure colon is removed here too
    
    return None

def refine_special_fields(data):
    """Refine medications and investigations with embeddings for medications"""
    branchId = data.get("branchId")
    organizationId = data.get("organizationId")
    
    # Add logging to debug
    logger.info(f"refine_special_fields - branchId: {branchId}, organizationId: {organizationId}")

    if "medications" in data and data["medications"]:
        refined_meds = []
        for med in data["medications"]:
            name = med.get("name", "")
            if name:
                refined = search_collection(medications_collection, name, use_embeddings=True, branchId=branchId, organizationId=organizationId)
                refined_meds.append(refined)
        
        # Deduplicate medications based on name
        seen_names = set()
        unique_meds = []
        for med in refined_meds:
            if med["name"].lower() not in seen_names:
                seen_names.add(med["name"].lower())
                unique_meds.append(med)
        data["medications"] = unique_meds
        
        interactions = check_drug_interactions(data["medications"])
        if interactions:
            data["drug_interactions"] = interactions
    
    if "investigations" in data and data["investigations"]:
        refined_invs = []
        for inv in data["investigations"]:
            name = inv.get("name", "")
            if name:
                refined = search_collection(investigations_collection, name, branchId=branchId, organizationId=organizationId)
                refined_invs.append(refined)
        data["investigations"] = refined_invs
    
    return data

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
            diagnosis_names = []
            for d in doctor_prefs['diagnosis'][:3]:
                if isinstance(d, dict):
                    diagnosis_names.append(d.get('name', ''))
                elif isinstance(d, str):
                    diagnosis_names.append(d)
            base_context += f"- Previous diagnoses: {', '.join(diagnosis_names)}\n"
        if doctor_prefs.get('investigations'):
            investigations = doctor_prefs['investigations'][:5] if len(doctor_prefs['investigations']) > 5 else doctor_prefs['investigations']
            base_context += f"- Previous investigations: {', '.join(investigations)}\n"
        if field == "medications" and doctor_prefs.get('medications'):
            medications = doctor_prefs['medications'][:3] if len(doctor_prefs['medications']) > 3 else doctor_prefs['medications']
            base_context += f"- Previous medications: {', '.join(medications)}\n"
        if doctor_prefs.get('diet_instructions') and doctor_prefs['diet_instructions']:
            diet_text = doctor_prefs['diet_instructions'][0][:50] if len(doctor_prefs['diet_instructions'][0]) > 50 else doctor_prefs['diet_instructions'][0]
            base_context += f"- Previous diet instructions: {diet_text}...\n"
        if doctor_prefs.get('followup') and doctor_prefs['followup']:
            followup_text = doctor_prefs['followup'][0].get('text', '')[:50] if len(doctor_prefs['followup'][0].get('text', '')) > 50 else doctor_prefs['followup'][0].get('text', '')
            base_context += f"- Previous followup: {followup_text}...\n"

    prompts = {
        "diagnosis": f"{base_context}\nProvide the most likely diagnosis based on current complaints.\nOutput format: [{{\"code\": \"ICD_CODE\", \"name\": \"DIAGNOSIS_NAME\"}}]\n{format_examples('diagnosis')}",
        "investigations": f"{base_context}\nProvide a concise, relevant list of investigations for the current condition.\nOutput format: [{{\"code\": \"\", \"name\": \"TEST_NAME\"}}]\n{format_examples('investigations')}",
        "medications": f"{base_context}\nProvide a list of unique medications (no duplicates) based on the patient's current complaints.\nOutput format: [{{\"code\": \"\", \"name\": \"MED_NAME\"}}]\n{format_examples('medications')}",
        "followup": f"{base_context}\nProvide a followup plan based on current condition with a date after March 20, 2025.\nOutput format: {{\"date\": \"YYYY-MM-DD\", \"text\": \"INSTRUCTIONS\"}}\n{format_examples('followup')}",
        "diet_instructions": f"{base_context}\nProvide concise diet instructions relevant to current condition as a single line.\nOutput format: \"INSTRUCTIONS\"\n{format_examples('diet_instructions')}"
    }
    return prompts.get(field, "")

def process_json_data(data):
    """Process input JSON and return completed data"""
    result = data.copy()
    patient_id = result.get("patientId")
    doctor_id = result.get("doctorId")
    branchId = result.get("branchId")
    organizationId = result.get("organizationId")
    
    if not patient_id or not doctor_id:
        logger.error("Missing patientId or doctorId")
        return {"error": "Missing patientId or doctorId"}, 400
    
    # Not updating patient details 
    chief_complaints = ""
    if "chief_complaints" in result:
        if isinstance(result["chief_complaints"], dict):
            chief_complaints = result["chief_complaints"].get("text", "")
        elif isinstance(result["chief_complaints"], str):
            chief_complaints = result["chief_complaints"]
        result["chief_complaints"] = [{"text": chief_complaints}]
        
    if not validate_chief_complaints(chief_complaints):
        logger.info(f"Invalid chief complaints: '{chief_complaints}'")
        return result
        
    doctor_prefs = fetch_doctor_preferences(doctor_id, chief_complaints)
    doctor_fields = ["diagnosis", "investigations", "medications", "followup", "diet_instructions"]
    
    for field in doctor_fields:
        is_empty = (field not in result or 
                    (isinstance(result[field], list) and not result[field]) or 
                    (isinstance(result[field], str) and not result[field].strip()) or 
                    (isinstance(result[field], dict) and not any(result[field].values())))
        if is_empty:
            prompt = generate_prompt(field, result, doctor_prefs)
            if prompt:
                llm_output = get_llm_suggestion(prompt)
                if llm_output:
                    processed_output = process_llm_output(llm_output, field)
                    if processed_output is not None:
                        if field == "diet_instructions" and isinstance(processed_output, str):
                            result[field] = [{"text": processed_output}]
                        else:
                            result[field] = processed_output
                        logger.info(f"Set {field} to: {result[field]}")
    
    # Check if diet_instructions is empty and set a default value
    if "diet_instructions" in result and (
        not result["diet_instructions"] or 
        (isinstance(result["diet_instructions"], list) and 
         (not result["diet_instructions"] or not result["diet_instructions"][0].get("text", "").strip()))
    ):
        result["diet_instructions"] = [{"text": "Stay hydrated, take plenty of fluids, and eat light, nutritious food."}]
        logger.info(f"Set default diet_instructions: {result['diet_instructions']}")
    
    if "diagnosis" in result and result["diagnosis"]:
        diagnosis_name = result["diagnosis"][0].get("name", "") if isinstance(result["diagnosis"], list) else result["diagnosis"].get("name", "")
        if diagnosis_name:
            api_diagnosis = {"name": diagnosis_name, "code": ""}
            result["diagnosis"] = [api_diagnosis]
    
    # Ensure branchId and organizationId are preserved
    result["branchId"] = branchId
    result["organizationId"] = organizationId
    
    # Use doctor preferences for medications more intelligently
    if doctor_prefs.get("medications") and doctor_prefs.get("scores") and doctor_prefs.get("conditions"):
        # Only consider using previous medications in very specific cases
        # Primarily rely on the LLM to generate appropriate medications for current condition
        existing_medications = [med["name"] for med in result["medications"]] if "medications" in result and result["medications"] else []
        
        # Check if we need to add a medication from past preferences
        if existing_medications:
            # Only look at the most similar past case with similarity over 0.7 (very similar)
            # Get indices of items with scores > 0.7
            high_similarity_indices = [i for i, score in enumerate(doctor_prefs["scores"]) if score > 0.7]
            
            if high_similarity_indices:
                # Extract medications with high similarity
                high_similarity_meds = [doctor_prefs["medications"][i] for i in high_similarity_indices]
                high_similarity_conditions = [doctor_prefs["conditions"][i] for i in high_similarity_indices]
                
                logger.info(" -----------  ------------  ------------")
                logger.info(f"Found {len(high_similarity_meds)} highly similar past medications")
                logger.info(f"Current complaints: {chief_complaints}")
                logger.info(f"Similar past conditions: {list(set(high_similarity_conditions))}")
                logger.info(" -----------  ------------  ------------")
                
                # Check for medications that would be relevant but aren't yet included
                # Limit to just one additional medication to avoid over-medication
                existing_meds_set = {med.lower() for med in existing_medications}
                for med in high_similarity_meds:
                    if med.lower() not in existing_meds_set:
                        # Add only one past medication and break
                        result["medications"].append({"code": "", "name": med})
                        logger.info(f"Added medication from past: {med}")
                        break
    
    result = refine_special_fields(result)
    
    # Remove keys we don't want
    keys_to_remove = ["medical_history", "personal_history", "family_history", 
                       "current_medications", "allergies"]
    for key in keys_to_remove:
        if key in result:
            del result[key]

    return result

@app.route('/api/suggest', methods=['POST'])
def suggest():
    """API endpoint to process input and provide suggestions"""
    try:
        data = request.json
        if not data or "patientId" not in data or "doctorId" not in data:
            return jsonify({
                "status": False,
                "data": {},
                "message": "Missing patientId or doctorId in request"
            }), 400
            
        completed_data = process_json_data(data)
        if isinstance(completed_data, tuple):
            return jsonify({
                "status": False,
                "data": completed_data[0],
                "message": "Invalid input data provided"
            }), completed_data[1]
        
        return jsonify({
            "status": True,
            "data": completed_data,
            "message": "Suggestions generated successfully"
        })
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({
            "status": False,
            "data": {},
            "message": f"Error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)