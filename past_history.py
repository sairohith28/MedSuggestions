from flask import Flask, request, jsonify
from pymongo import MongoClient
import certifi
from bson import ObjectId
import requests
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["chatbot_dev"]
patient_visits_collection = db["patient-visits"]

# LLM API endpoint and authorization
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_AUTH_HEADER = os.getenv("LLM_AUTH_HEADER")

def fetch_patient_history(patient_id):
    """Fetch last 10 consultations for patient"""
    try:
        visits = list(patient_visits_collection.find({"patientId": ObjectId(patient_id)}).sort("createdAt", -1).limit(10))
        return visits
    except Exception as e:
        logger.error(f"Error fetching patient history: {str(e)}")
        return []

def generate_medical_history_summary(patient_id, org_id):
    """Generate a medical history summary for a patient"""
    # Fetch patient history
    past_visits = fetch_patient_history(patient_id)
    
    if not past_visits:
        return ""
    
    # Extract data needed for summarization
    past_10_medical_histories = ""
    past_10_chief_complaints = ""
    
    for visit in past_visits:
        visit_date = visit.get('createdAt', '')
        date_str = visit_date.strftime('%Y-%m-%d') if isinstance(visit_date, datetime) else ""
        
        # Format medical history
        if isinstance(visit.get("medical_history"), dict):
            past_10_medical_histories += f"Date: {date_str}\n"
            past_10_medical_histories += visit.get("medical_history", {}).get("content", "") + "\n\n"
        else:   
            past_10_medical_histories += f"Date: {date_str}\n"
            past_10_medical_histories += visit.get("medical_history", "") + "\n\n"
        
        # Format chief complaints
        if isinstance(visit.get("chief_complaints"), dict):
            past_10_chief_complaints += f"Date: {date_str}\n"
            past_10_chief_complaints += visit.get("chief_complaints", {}).get("content", "") + "\n\n"
        else:
            past_10_chief_complaints += f"Date: {date_str}\n"
            past_10_chief_complaints += visit.get("chief_complaints", "") + "\n\n"
    
    # Create prompt for medical history summarization with stronger enforcement of format
    medical_history_summarizer_prompt = """
    You are a specialized healthcare data analyst tasked with creating strict date-prefixed summaries.

    FORMAT REQUIREMENT: Your output MUST follow this exact format:
    "YYYY-MM-DD: Patient information. YYYY-MM-DD: More patient information."

    Every piece of information MUST start with a date in YYYY-MM-DD format followed by a colon.
    
    Example of CORRECT format:
    "2023-01-01: Patient had fever and cough. 2022-12-01: Patient had surgery."
    
    Example of INCORRECT format:
    "Patient had fever on 2023-01-01. Surgery was performed on 2022-12-01."
    "Patient reports symptoms for the past week."
    
    Your task:
    1. Analyze the chief complaints and medical histories
    2. Summarize in exactly the format: "YYYY-MM-DD: Information. YYYY-MM-DD: Information."
    3. Total summary must be <15 words (KEEP IT VERY SHORT)
    4. Use ONLY the exact dates provided in the "Date:" fields
    5. Do NOT create or infer dates - use ONLY the exact dates given
    6. STOP after 2-3 date entries to keep it short
    
    INPUT DATA:
    Chief complaints:
    {past_10_chief_complaints}
    
    Medical histories:
    {past_10_medical_histories}
    
    IMPORTANT RULES:
    - KEEP IT VERY SHORT - MAXIMUM 15 WORDS
    - EVERY piece of information MUST start with "YYYY-MM-DD: "
    - NO narrative text without dates
    - NO interpretation of timeframes (like "past week")
    - ONLY output the summary in the required format - no explanations
    - If no meaningful data exists, return an empty string
    - NO summary headers or prefixes (like "Summary:")
    - Dates refer to when information was recorded, NOT when conditions occurred
    
    OUTPUT FORMAT:
    "YYYY-MM-DD: Information. YYYY-MM-DD: Information."
    """
    
    medical_history_summarizer_prompt = medical_history_summarizer_prompt.format(
        past_10_chief_complaints=past_10_chief_complaints,
        past_10_medical_histories=past_10_medical_histories
    )
    
    # Call LLM API to summarize the medical history
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a specialized healthcare data analyst. Your ONLY task is to format medical data with date prefixes in the EXACT format 'YYYY-MM-DD: Information' for each data point. Keep your response very short (2-3 entries only)."
            },
            {"role": "user", "content": medical_history_summarizer_prompt}
        ]
        
        response = requests.post(
            LLM_API_URL,
            json={
                "model": "unsloth/Qwen2.5-1.5B-Instruct",
                "messages": messages,
                "temperature": 0.1,  # Lower temperature for more deterministic outputs
                "max_tokens": 150  # Reduced max tokens to prevent long outputs
            },
            headers={"Authorization": LLM_AUTH_HEADER},
        )
        
        response.raise_for_status()
        response_body = response.json()
        summary = response_body["choices"][0]["message"]["content"].strip('" \n')
        
        # Modified validation - check if it starts with a date format, but don't reject the entire output
        if summary:
            first_part = summary.split('. ')[0]
            if not (first_part.startswith('20') and '-' in first_part[:10] and ': ' in first_part[:13]):
                logger.warning(f"LLM output not in correct format: {summary}")
                # Try to extract a valid portion
                date_entries = []
                for part in summary.split('. '):
                    if part.startswith('20') and '-' in part[:10] and ': ' in part[:13]:
                        date_entries.append(part)
                
                if date_entries:
                    return '. '.join(date_entries) + '.'
                return ""
            
            # Fix cases where the output might be truncated at the end
            if not summary.endswith('.'):
                last_complete_sentence = '.'.join(summary.split('.')[:-1])
                if last_complete_sentence:
                    summary = last_complete_sentence + '.'
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating medical history summary: {str(e)}")
        return ""

@app.route('/api/patient_medical_history', methods=['POST'])
def get_patient_medical_history():
    """API endpoint to return medical history summary for a patient"""
    try:
        data = request.json
        
        # Validate required fields
        if not data or "patient_id" not in data:
            return jsonify({
                "status": False,
                "data": "",
                "message": "Missing patient_id in request"
            }), 400
            
        patient_id = data.get("patient_id")
        org_id = data.get("org_id", "")  # Optional parameter
        
        # Generate medical history summary
        summary = generate_medical_history_summary(patient_id, org_id)
        
        return jsonify({
            "status": True,
            "data": summary,
            "message": "Medical history summary generated successfully"
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({
            "status": False,
            "data": "",
            "message": f"Error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)