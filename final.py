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
    valid_levels = {"Minor", "Moderate", "Major"}
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
    """Fetch doctor preferences using semantic similarity"""
    if not chief_complaints or not isinstance(chief_complaints, str) or not chief_complaints.strip():
        return {}
    
    try:
        visits = patient_visits_collection.find({"doctorId": ObjectId(doctor_id)}).sort("createdAt", -1).limit(50)
        doctor_visits = list(visits)
        
        preferences = {"diagnosis": [], "investigations": [], "medications": [], "diet_instructions": [], "followup": [], "scores":[]}
        current_embedding = semantic_model.encode(chief_complaints.lower(), convert_to_tensor=True)
        
        for visit in doctor_visits:
            visit_complaints = visit.get("chief_complaints", {}).get("content", "").lower() if isinstance(visit.get("chief_complaints"), dict) else visit.get("chief_complaints", "").lower()
            if not visit_complaints:
                continue
                
            visit_embedding = semantic_model.encode(visit_complaints, convert_to_tensor=True)
            similarity = util.cos_sim(current_embedding, visit_embedding).item()
            
            if similarity > 0.5:  #change-1-musadiq 0.4 to 0.7
                if "diagnosis" in visit and visit["diagnosis"]:
                    preferences["diagnosis"].extend(visit["diagnosis"])
                if "investigations" in visit and visit["investigations"]:
                    investigations = [item["name"] for item in visit["investigations"]] if isinstance(visit["investigations"], list) else visit["investigations"]
                    preferences["investigations"].extend(investigations)
                if "medications" in visit and visit["medications"]:
                    medications = [item["name"] for item in visit["medications"]] if isinstance(visit["medications"], list) else visit["medications"]
                    preferences["medications"].extend(medications)
                    if isinstance(visit["medications"], list):
                        preferences['scores'].extend([similarity]*len(visit['medications']))
                    else:
                        preferences['scores'].append(similarity)
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
    print("starting extract patient details")
    details = {
        "medical_history": [{"text": ""}],
        "personal_history": [{"text": ""}],
        "family_history": [{"text": ""}],
        "current_medications": [{"text": ""}],
        "allergies": [{"code": "", "name": ""}]
    }
    
    if not past_visits:
        return details

    latest_visit = past_visits[0]
    print(past_visits)
    past_10_medical_histories = ""
    past_10_chief_complaints = ""
    past_10_allergies = ""
    print("before loop")
    for i in past_visits:
        if isinstance(i.get("medical_history"), dict):
            past_10_medical_histories += f"Date: {i.get('createdAt', '').strftime('%Y-%m-%d')}\n"
            past_10_medical_histories += i.get("medical_history", {}).get("content", "") + "\n"
        else:   
            past_10_medical_histories += f"Date: {i.get('createdAt', '').strftime('%Y-%m-%d')}\n"
            past_10_medical_histories += i.get("medical_history", "") + "\n"
        if isinstance(i.get("chief_complaints"), dict):
            past_10_chief_complaints += f"Date: {i.get('createdAt', '').strftime('%Y-%m-%d')}\n"
            past_10_chief_complaints += i.get("chief_complaints", {}).get("content", "") + "\n"
        else:
            past_10_chief_complaints += f"Date: {i.get('createdAt', '').strftime('%Y-%m-%d')}\n"
            past_10_chief_complaints += i.get("chief_complaints", "") + "\n"
        if isinstance(i.get("allergies"), dict):
            past_10_allergies += i.get("allergies", {}).get("content", "") + "\n"
        else:
            past_10_allergies += i.get("allergies", "") + "\n"
    medical_history_summarizer_prompt = """
        You act as a specalized healthcare data analyst. You're provided with last 10 consultations of a patient.
        Your have to analyze chief complaints and past medical histories of patient and summarize a paragraph with less than 30 words based on below requirement.
        Requirement:
        You have to analyze the previous chief complaints and past medical histories of patient and summarize a paragraph indicating dates too with less than 30 words  for the doctor understanding of the patient in the past.
        If there are not any previous medical histories or chief complaints, don't generate any. just an empty string.
        You have to include the following information in the summary:
        1. Chief complaints of patient
        2. Past medical histories of patient
        Here are your inputs
        Chief complaints:\n {past_10_chief_complaints}
        \n
        Past medical histories:\n {past_10_medical_histories}
        You have to summarize the above information in a single paragraph with less than 30 words. And strictly don't hallucinate on any patient information and don't provide any misinformation.
        Remember to include the dates in the summary which were provided to you by DATE: key in the input. which helps doctors to understand the patient history in the past timeframes.
        Ex: "2023-01-01: Patient had fever and cough. 2022-12-01: Patient had surgery"
        Don't use relative timeframes like "last week" or "yesterday" which were used by the doctor on that particular day. Because those were used to write in the past, those timeframes cannot be used in generating summary now. 
        The dates in the information provided to you doesn't indicate the date at which patient got that disease or surgery or any procedure. It indicates the date at which doctor has written that information in the clinical note. So don't use those dates as the date at which patient got that disease or surgery or any procedure. 
        Use the dates to indicate the date at which doctor has written that information in the clinical note. So don't use those dates as the date at which patient got that disease or surgery or any procedure.
        Striclty use dates while generating summary
        And strictly don't include any tags before generating the summary like "SUMMARY: " or "Output: ". Just a plain string with the summary.
        Don't include any other information or explanation. Just the summary.
        Don't use timeframes specified in the 
        Strictly analyze and use dates as the date which that particular event is noted in the summary. Don't generate the summary without the dates. And don't use new line between the events in the summary.
        Stricly always indicate dates for the events of procedures or medical histories as a note that doctor will know when this happened. use the example provided above while generating the summary with dates.
        Output:
    """
    medical_history_summarizer_prompt = medical_history_summarizer_prompt.format(
        past_10_chief_complaints=past_10_chief_complaints,
        past_10_medical_histories=past_10_medical_histories
    )
    print("final prompt: ", medical_history_summarizer_prompt)
    messages = [
        {
            "role": "system",
            "content": "You act as a specalized healthcare data analyst and do whatever is in the prompt",
        },
        {"role": "user", "content": medical_history_summarizer_prompt},
    ]
    medication_response = requests.post(
            "http://203.112.158.104:5006/v1/chat/completions",
            json={
                "model": "unsloth/Qwen2.5-1.5B-Instruct",
                "messages": messages,
            },
            headers={"Authorization": "Bearer apex@#1"},
        )
    allergy_summarizer_prompt = """
        You act as a specalized healthcare data analyst. You're provided with last 10 consultations of a patient.
        Your have to analyze allergies of patient and summarize a paragraph with less than 30 words based on below requirement.
        Requirement:
        You have to analyze the previous allergies of patient and summarize a paragraph with less than 30 words for the doctor understanding of the patient in the past.
        If there are not any previous allergies, don't generate any. just an empty string.
        Here are your inputs
        Allergies: {past_10_allergies}
        You have to summarize the above information in a single paragraph with less than 30 words. And strictly don't hallucinate on any patient information and don't provide any misinformation.
        And strictly don't include any tags before generating the summary like "SUMMARY: " or "Output: ". Just a plain string with the summary.
        Ex: "Patient had allergy to penicillin"
        Don't use relative timeframes like "last week" or "yesterday" which were used by the doctor on that particular day. Because those were used to write in the past, those timeframes cannot be used in generating summary now. 
        Output:
    """
    allergy_summarizer_prompt = allergy_summarizer_prompt.format(
        past_10_allergies=past_10_allergies
    )
    messages = [
        {
            "role": "system",
            "content": "You act as a specalized healthcare data analyst and do whatever is in the prompt",
        },
        {"role": "user", "content": allergy_summarizer_prompt},
    ]
    allergy_response = requests.post(
                "http://203.112.158.104:5006/v1/chat/completions",
                json={
                    "model": "unsloth/Qwen2.5-1.5B-Instruct",
                    "messages": messages,
                },
                headers={"Authorization": "Bearer apex@#1"},
            )
    medication_response_body = medication_response.json()
    allergy_response_body = allergy_response.json()
    details["medical_history"] = [{"text": medication_response_body["choices"][0]["message"]["content"]}]
    details["personal_history"] = [{"text": latest_visit.get("personal_history", {}).get("content", "") if isinstance(latest_visit.get("personal_history"), dict) else latest_visit.get("personal_history", "")}]
    details["family_history"] = [{"text": latest_visit.get("family_history", {}).get("content", "") if isinstance(latest_visit.get("family_history"), dict) else latest_visit.get("family_history", "")}]
    details["current_medications"] = [{"text": latest_visit.get("current_medications", {}).get("content", "") if isinstance(latest_visit.get("current_medications"), dict) else latest_visit.get("current_medications", "")}]
    
    # allergies = latest_visit.get("allergies", [])
    # if isinstance(allergies, list) and allergies:
    #     details["allergies"] = [{"code": str(item.get("code", "")), "name": item.get("name", "")} for item in allergies]
    # elif isinstance(allergies, str) and allergies:
    #     details["allergies"] = [{"code": "", "name": allergies}]
    details["allergies"] = [{"code": "", "name": allergy_response_body["choices"][0]["message"]["content"]}]
    
    return details

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
    print("generate prompt calling")
    chief_complaints_list = patient_data.get("chief_complaints", [{"text": ""}])
    chief_complaints = chief_complaints_list[0].get("text", "") if chief_complaints_list else ""
    print("chief complaints done")
    # Get medical history
    medical_history_list = patient_data.get("medical_history", [{"text": ""}])
    medical_history = medical_history_list[0].get("text", "") if medical_history_list else ""
    
    # Get allergies
    allergies_list = patient_data.get("allergies", [{"name": ""}])
    allergies = ", ".join([allergy.get("name", "") for allergy in allergies_list if allergy.get("name")])
    print("allergies done")
    if not chief_complaints or not validate_chief_complaints(chief_complaints):
        print("is empty checking")
        return ""

    base_context = f"""
Patient Information:
- Chief complaints: {chief_complaints}
- Medical history: {medical_history}
- Allergies: {allergies}
"""

    if doctor_prefs:
        base_context += "Doctor Preferences (use as reference only, prioritize current condition):\n"
        if doctor_prefs.get('diagnosis'):
            print("doctor prefs diagnosis")
            diagnosis_names = []
            for d in doctor_prefs['diagnosis'][:3]:
                if isinstance(d, dict):
                    diagnosis_names.append(d.get('name', ''))
                elif isinstance(d, str):
                    diagnosis_names.append(d)
            base_context += f"- Previous diagnoses: {', '.join(diagnosis_names)}\n"
            print("doctor prefs diagnosis done")
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
    print("doctor prefs done")
    prompts = {
        "diagnosis": f"{base_context}\nProvide the most likely diagnosis based on current complaints and history.\nOutput format: [{{\"code\": \"ICD_CODE\", \"name\": \"DIAGNOSIS_NAME\"}}]\n{format_examples('diagnosis')}",
        "investigations": f"{base_context}\nProvide a concise, relevant list of investigations for the current condition.\nOutput format: [{{\"code\": \"\", \"name\": \"TEST_NAME\"}}]\n{format_examples('investigations')}",
        "medications": f"{base_context}\nProvide a list of unique medications (no duplicates) based on the patient's current complaints and history.\nOutput format: [{{\"code\": \"\", \"name\": \"MED_NAME\"}}]\n{format_examples('medications')}",
        "followup": f"{base_context}\nProvide a followup plan based on current condition with a date after March 20, 2025.\nOutput format: {{\"date\": \"YYYY-MM-DD\", \"text\": \"INSTRUCTIONS\"}}\n{format_examples('followup')}",
        "diet_instructions": f"{base_context}\nProvide concise diet instructions relevant to current condition as a single line.\nOutput format: \"INSTRUCTIONS\"\n{format_examples('diet_instructions')}"
    }
    print("prompts done")
    return prompts.get(field, "")

def process_json_data(data):
    """Process input JSON and return completed data"""
    print("starting process json data")
    result = data.copy()
    patient_id = result.get("patientId")
    doctor_id = result.get("doctorId")
    branchId = result.get("branchId")
    organizationId = result.get("organizationId")
    
    if not patient_id or not doctor_id:
        logger.error("Missing patientId or doctorId")
        return {"error": "Missing patientId or doctorId"}, 400
    print("before extract patient details")
    past_visits = fetch_patient_history(patient_id)
    patient_details = extract_patient_details(past_visits)
    result.update(patient_details)
    print("extraction done")
    chief_complaints = ""
    if "chief_complaints" in result:
        if isinstance(result["chief_complaints"], dict):
            chief_complaints = result["chief_complaints"].get("text", "")
        elif isinstance(result["chief_complaints"], str):
            chief_complaints = result["chief_complaints"]
        result["chief_complaints"] = [{"text": chief_complaints}]
    medical_history = ""
    if "medical_history" in result and result["medical_history"]:
        if isinstance(result["medical_history"], list) and result["medical_history"]:
            medical_history = result["medical_history"][0].get("text", "")
        elif isinstance(result["medical_history"], str):
            medical_history = result["medical_history"]

    allergies = ""
    if "allergies" in result and result["allergies"]:
        if isinstance(result["allergies"], list):
            allergies = ", ".join([allergy.get("name", "") for allergy in result["allergies"] if allergy.get("name")])
        elif isinstance(result["allergies"], str):
            allergies = result["allergies"]
    if not validate_chief_complaints(chief_complaints):
        logger.info(f"Invalid chief complaints: '{chief_complaints}'")
        return result
    print("updation done")
    doctor_prefs = fetch_doctor_preferences(doctor_id, chief_complaints)
    print("fetching doctor done")
    doctor_fields = ["diagnosis", "investigations", "medications", "followup", "diet_instructions"]
    for field in doctor_fields:
        is_empty = (field not in result or 
                    (isinstance(result[field], list) and not result[field]) or 
                    (isinstance(result[field], str) and not result[field].strip()) or 
                    (isinstance(result[field], dict) and not any(result[field].values())))
        if is_empty:
            print("is empty checking")
            prompt = generate_prompt(field, result, doctor_prefs)
            print("generate prompt done")
            if prompt:
                llm_output = get_llm_suggestion(prompt)
                print("llm suggestion done")
                if llm_output:
                    processed_output = process_llm_output(llm_output, field)
                    if processed_output is not None:
                        if field == "diet_instructions" and isinstance(processed_output, str):
                            result[field] = [{"text": processed_output}]
                        else:
                            result[field] = processed_output
                        logger.info(f"Set {field} to: {result[field]}")
    print("after loop doctor_fields")
    if "diagnosis" in result and result["diagnosis"]:
        diagnosis_name = result["diagnosis"][0].get("name", "") if isinstance(result["diagnosis"], list) else result["diagnosis"].get("name", "")
        if diagnosis_name:
            # api_diagnosis = get_icd11_diagnosis(diagnosis_name)
            api_diagnosis = {"name":diagnosis_name, "code":""}
            if api_diagnosis:
                result["diagnosis"] = [api_diagnosis]
    print("after loop doctor_fields")
    print("res: ", result)
    if "medical_history" in result:
        print("--------------- medical history", result["medical_history"])
    # Ensure branchId and organizationId are preserved
    result["branchId"] = branchId
    result["organizationId"] = organizationId
    logger.info(f"process_json_data - Passing to refine_special_fields - branchId: {branchId}, organizationId: {organizationId}")
    
    #change - 2 - musadiq added lines 10598 to 10603
    # if doctor_prefs.get("medications"):
    #     existing_meds = set([d["name"] for d in result["medications"]])
    #     for med in doctor_prefs["medications"]:
    #         if med not in existing_meds:
    #             result["medications"].append({"code": "", "name": med})
    #             existing_meds.add(med)  # Keep the set updated
    if doctor_prefs.get("medications") and doctor_prefs.get("scores"):
        # max_score = max(doctor_prefs["scores"])
        # Get medications only from the visit with the highest similarity
        # highest_pref_medications = []
        # for i in range(len(scores)):
        #     if scores[i] == max_score:
        #         highest_pref_medications.append(doctor_prefs['medications'][i])
        highest_pref_medications = [
        med for med, _ in sorted(
            zip(doctor_prefs["medications"], doctor_prefs["scores"]),
            key=lambda x: x[1],
            reverse=True
        )
    ]   
        logger.info(" -----------  ------------  ------------")
        logger.info("The Highest Prefered medications are ", highest_pref_medications)
        logger.info(" -----------  ------------  ------------")
        existing_meds = {d["name"] for d in result["medications"]}
        for med in highest_pref_medications:
            if med not in existing_meds:
                result["medications"].append({"code": "", "name": med})
                # existing_meds.add(med)
                break
  
    
    result = refine_special_fields(result)

    return result

@app.route('/api/suggest', methods=['POST'])
def suggest():
    """API endpoint to process input and provide suggestions"""
    print("starting /api/suggest")
    try:
        data = request.json
        if not data or "patientId" not in data or "doctorId" not in data:
            return jsonify({
                "status": False,
                "data": {},
                "message": "Missing patientId or doctorId in request"
            }), 400
        print(f"calling proces data function with data", data)
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