import json
import requests
from flask import Flask, request, jsonify
import logging
import re
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LLM API endpoint and authorization
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_AUTH_HEADER = os.getenv("LLM_AUTH_HEADER")

# Example cases for few-shot learning
EXAMPLE_CASES = {
    "diagnosis": [
        {
            "input": "Chief complaints: Persistent dry cough and mild dyspnea for one week, no fever. Allergies: Dust and pollen. Medical history: Seasonal allergic rhinitis.",
            "output": [{"code": "J45.909", "name": "Allergic asthma"}]
        },
        {
            "input": "Chief complaints: Severe headache, photophobia, nausea for 2 days. Medical history: Migraine sufferer since adolescence.",
            "output": [{"code": "G43.909", "name": "Migraine, unspecified, not intractable"}]
        },
        {
            "input": "Chief complaints: High fever, productive cough with yellow sputum, chest pain. Medical history: COPD.",
            "output": [{"code": "J15.9", "name": "Bacterial pneumonia"}]
        }
    ],
    "investigations": [
        {
            "input": "Chief complaints: Persistent dry cough and mild dyspnea for one week, no fever. Allergies: Dust and pollen. Medical history: Seasonal allergic rhinitis.",
            "output": "CBC, CRP, spirometry, chest X-ray, IgE levels"
        },
        {
            "input": "Chief complaints: High knee pain. Medical history: Previous knee injury 2 years ago.",
            "output": "X-ray knee, CBC, ESR, CRP, Knee MRI"
        },
        {
            "input": "Chief complaints: Recurrent abdominal pain, bloating, altered bowel habits. Medical history: Family history of colon cancer.",
            "output": "CBC, CRP, fecal occult blood test, colonoscopy, abdominal ultrasound"
        }
    ],
    "medications": [
        {
            "input": "Diagnosis: Allergic asthma. Chief complaints: Persistent dry cough and mild dyspnea.",
            "output": [
                {"name": "Levosalbutamol", "instructions": "Take one 1 mg tablet at night for five days"},
                {"name": "Montelukast", "instructions": "Take one 10 mg tablet at night for five days"},
                {"name": "Budesonide", "instructions": "Inhale 100 mcg twice daily for one week"}
            ]
        },
        {
            "input": "Diagnosis: Osteoarthritis of knee. Chief complaints: High knee pain.",
            "output": [
                {"name": "Diclofenac", "instructions": "Take one 50 mg tablet twice daily with food for 7 days"},
                {"name": "Paracetamol", "instructions": "Take two 500 mg tablets every 6 hours as needed for pain"},
                {"name": "Glucosamine", "instructions": "Take one 500 mg tablet twice daily with meals"}
            ]
        },
        {
            "input": "Diagnosis: Hypertension. Chief complaints: Headache, dizziness. Vitals: BP 160/95 mmHg.",
            "output": [
                {"name": "Amlodipine", "instructions": "Take one 5 mg tablet once daily in the morning"},
                {"name": "Hydrochlorothiazide", "instructions": "Take one 12.5 mg tablet once daily in the morning"}
            ]
        }
    ],
    "followup": [
        {
            "input": "Diagnosis: Allergic asthma. Chief complaints: Persistent dry cough and mild dyspnea.",
            "output": {"date": "2025-03-10", "text": "Review in 2 weeks with spirometry results. Return sooner if symptoms worsen."}
        },
        {
            "input": "Diagnosis: Osteoarthritis of knee. Chief complaints: High knee pain.",
            "output": {"date": "2025-03-20", "text": "Follow up in 3 weeks with X-ray and MRI reports. Continue prescribed physical therapy."}
        },
        {
            "input": "Diagnosis: Hypertension. Vitals: BP 160/95 mmHg.",
            "output": {"date": "2025-03-05", "text": "Review in 1 week for blood pressure check and medication adjustment if needed."}
        }
    ],
    "diet_instructions": [
        {
            "input": "Diagnosis: Allergic asthma. Chief complaints: Persistent dry cough and mild dyspnea. Allergies: Dust and pollen.",
            "output": "Increase hydration, avoid exposure to dust, dairy, and cold beverages, and consume antioxidant-rich foods"
        },
        {
            "input": "Diagnosis: Osteoarthritis of knee. Chief complaints: High knee pain.",
            "output": "Maintain a balanced diet, increase anti-inflammatory foods like turmeric and fish, avoid processed foods and excess sugar"
        },
        {
            "input": "Diagnosis: Hypertension. Chief complaints: Headache, dizziness. Vitals: BP 160/95 mmHg.",
            "output": "Reduce salt intake, avoid caffeine and alcohol, increase potassium-rich foods like bananas and leafy greens"
        }
    ]
}

def format_examples(field):
    """Format examples for few-shot learning for a specific field"""
    examples = EXAMPLE_CASES.get(field, [])
    formatted = ""
    
    for i, example in enumerate(examples):
        formatted += f"\nExample {i+1}:\n"
        formatted += f"Patient information: {example['input']}\n"
        formatted += f"Response: {json.dumps(example['output'], ensure_ascii=False)}\n"
    
    return formatted

def get_llm_suggestion(prompt):
    """
    Get suggestions from the medical LLM.
    """
    headers = {
        "Authorization": LLM_AUTH_HEADER,
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "unsloth/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are an expert Medical AI assistant specialized in providing structured medical information. Always follow the specified output format exactly without explanation or reasoning. Be concise and precise."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.2,
        "seed": 42  
    }
    
    try:
        response = requests.post(LLM_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            logger.error(f"Unexpected response format: {result}")
            return None
    except Exception as e:
        logger.error(f"Error calling LLM API: {str(e)}")
        return None

def get_icd11_diagnosis(diagnosis_name):
    """
    Fetch diagnosis name and ICD-11 code from the provided API based on diagnosis name.
    """
    try:
        url = f"https://dev.apexcura.com/api/op/getDiagnosisList?searchText={diagnosis_name}"
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        
        if result.get("status") and result.get("data") and len(result["data"]) > 0:
            top_result = result["data"][0]
            return {"name": top_result["diagnosis"], "code": top_result["icdCode"]}
        return None
    except Exception as e:
        logger.error(f"Error fetching ICD-11 diagnosis: {str(e)}")
        return None

def predict_allergies(patient_data):
    """
    Predict potential allergies based on patient data.
    """
    allergies = patient_data.get('allergies', 'Not provided')
    if allergies != 'Not provided':
        return allergies
    
    complaints = patient_data.get('chief_complaints', '').lower()
    history = patient_data.get('medical_history', '').lower()
    predicted = []
    
    if 'cough' in complaints or 'dyspnea' in complaints or 'rhinitis' in history:
        predicted.append("Dust")
        predicted.append("Pollen")
    if 'rash' in complaints or 'dermatitis' in history:
        predicted.append("Nickel")
    if 'abdominal pain' in complaints or 'diarrhea' in complaints:
        predicted.append("Lactose")
    
    return ", ".join(predicted) if predicted else "None predicted"

def extract_json_from_text(text):
    """
    Attempts to extract JSON from text that might contain explanations or other content.
    """
    json_pattern = r'\[.*\]|\{.*\}'
    match = re.search(json_pattern, text, re.DOTALL)
    
    if match:
        try:
            json_text = match.group(0)
            return json.loads(json_text)
        except:
            pass
    
    return None

def generate_field_prompt(patient_data, field):
    """
    Generate prompt for a specific field based on available patient data with examples.
    """
    base_context = f"""
Based on the following patient information:
- Chief complaints: {patient_data.get('chief_complaints', 'Not provided')}
- Allergies: {patient_data.get('allergies', 'Not provided')}
- Medical history: {patient_data.get('medical_history', 'Not provided')}
- Current medications: {patient_data.get('current_medications', 'Not provided')}
- Vitals: {patient_data.get('vitals', 'Not provided')}
"""
    
    if field in ["medications", "diet_instructions"] and "diagnosis" in patient_data and patient_data["diagnosis"]:
        if isinstance(patient_data["diagnosis"], list) and len(patient_data["diagnosis"]) > 0:
            diag = patient_data["diagnosis"][0].get("name", "")
            base_context += f"- Diagnosis: {diag}\n"

    prompts = {
        "diagnosis": base_context + f"""
I need you to provide ONLY the most likely diagnosis with an appropriate ICD code based on the patient's information.

Your output MUST follow this exact format without any explanation or additional text:
[{{"code": "ICD_CODE", "name": "DIAGNOSIS_NAME"}}]

The diagnosis name should be concise (2-4 words) and the ICD code should be valid. Do not include any reasoning or explanation in your response.

{format_examples("diagnosis")}

Now, provide ONLY the diagnosis in the format described above:
""",

        "investigations": base_context + f"""
I need you to provide ONLY a concise list of recommended investigations/tests for this patient.

Your output MUST be a comma-separated list of test names without any explanation or additional text.
The list should be specific, relevant to the patient's condition, and no more than 5-7 items.

{format_examples("investigations")}

Now, provide ONLY the list of investigations as described above:
""",

        "medications": base_context + f"""
I need you to provide ONLY a list of recommended medications with clear dosage instructions for this patient.

Your output MUST follow this exact format without any explanation or additional text:
[
  {{"name": "MEDICATION_NAME", "instructions": "PRECISE_DOSAGE_AND_INSTRUCTIONS"}},
  {{"name": "MEDICATION_NAME", "instructions": "PRECISE_DOSAGE_AND_INSTRUCTIONS"}}
]

- Each medication name should be a real drug name without explanations
- Instructions should include dosage, frequency, duration, and any special instructions
- Provide 2-4 appropriate medications for the patient's condition

{format_examples("medications")}

Now, provide ONLY the medications list in the format described above:
""",

        "followup": base_context + f"""
I need you to provide ONLY a concise followup plan with a date and instructions.

Your output MUST follow this exact format without any explanation or additional text:
{{"date": "YYYY-MM-DD", "text": "BRIEF_FOLLOWUP_INSTRUCTIONS"}}

- The date should be 1-4 weeks in the future depending on the condition severity
- The text should be a single, clear sentence about when to return and what to bring
- Do not include any reasoning or explanation in your response

{format_examples("followup")}

Now, provide ONLY the followup plan in the format described above:
""",

        "diet_instructions": base_context + f"""
I need you to provide ONLY a concise set of diet instructions for this patient.

Your output MUST be a single string without any explanation or additional text.
The instructions should be practical, relevant to the patient's condition, and 15-30 words long.

{format_examples("diet_instructions")}

Now, provide ONLY the diet instructions as a single string:
"""
    }
    
    return prompts.get(field, "")

def process_field(field, suggestion):
    """
    Process field-specific suggestion and convert to required format.
    """
    if not suggestion:
        return None
        
    try:
        json_content = extract_json_from_text(suggestion)
        if json_content:
            return json_content
            
        if field == "diagnosis":
            match = re.search(r'([A-Z][0-9]+\.[0-9]+)', suggestion)
            code = match.group(1) if match else "Unknown"
            name = suggestion.split('.')[0].strip()
            if len(name) > 100:
                name = name[:100] + "..."
            return [{"code": code, "name": name}]
            
        elif field == "investigations":
            return suggestion.strip().replace("\n", ", ")
            
        elif field == "medications":
            lines = suggestion.strip().split("\n")
            meds = []
            for line in lines:
                if ":" in line:
                    name, instructions = line.split(":", 1)
                    meds.append({"name": name.strip(), "instructions": instructions.strip()})
            if meds:
                return meds
            parts = suggestion.split(".", 1)
            return [{"name": parts[0].strip(), "instructions": parts[1].strip() if len(parts) > 1 else "As directed"}]
                
        elif field == "followup":
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', suggestion)
            date = date_match.group(1) if date_match else "2025-03-15"
            text = suggestion.replace(date, "").strip() if date_match else suggestion.strip()
            if len(text) > 200:
                text = text.split(".")[0] + "."
            return {"date": date, "text": text}
            
        elif field == "diet_instructions":
            return suggestion.strip()
            
        else:
            return suggestion.strip()
            
    except Exception as e:
        logger.error(f"Error processing field {field}: {str(e)}")
        return suggestion.strip()

def process_json_data(data):
    """
    Process the input JSON data and return completed data with AI suggestions.
    """
    result = data.copy() if data else {}
    
    # Fields to check and potentially fill
    fields_to_check = ["diagnosis", "investigations", "medications", "followup", "diet_instructions"]
    
    # Predict allergies if not provided or needs fixing
    if "allergies" not in data or data["allergies"] == "" or "predicted" in data["allergies"].lower():
        result["allergies"] = predict_allergies(data)
    
    for field in fields_to_check:
        field_empty = (field not in data or not data[field] or 
                      (isinstance(data.get(field), str) and data[field].strip() == "") or
                      (isinstance(data.get(field), list) and len(data[field]) == 0) or
                      (isinstance(data.get(field), dict) and 
                       all(not v for v in data[field].values())))
                       
        field_needs_fixing = False
        if field in data:
            if field == "diagnosis" and isinstance(data.get(field), list) and len(data[field]) > 0:
                diag_name = data[field][0].get("name", "")
                if len(diag_name) > 50 or diag_name.lower().startswith("to determine"):
                    field_needs_fixing = True
            elif field == "medications" and isinstance(data.get(field), list) and len(data[field]) > 0:
                for med in data[field]:
                    if not isinstance(med, dict) or ("I'm confident" in med.get("name", "")):
                        field_needs_fixing = True
                        break
            elif field == "followup" and isinstance(data.get(field), dict):
                followup_text = data[field].get("text", "")
                if followup_text.startswith("Alright, let's think") or "step by step" in followup_text.lower():
                    field_needs_fixing = True
            elif field == "diet_instructions" and isinstance(data.get(field), str):
                if "thinking" in data[field].lower() or len(data[field]) > 100:
                    field_needs_fixing = True
                       
        if field_empty or field_needs_fixing:
            logger.info(f"Field '{field}' is missing, empty, or needs fixing. Generating suggestion.")
            
            prompt = generate_field_prompt(result, field)
            if not prompt:
                continue
                
            suggestion = get_llm_suggestion(prompt)
            if not suggestion:
                continue
                
            processed_value = process_field(field, suggestion)
            if processed_value is not None:
                result[field] = processed_value
    
    # Update diagnosis with API-provided name and ICD-11 code
    if "diagnosis" in result and isinstance(result["diagnosis"], list) and len(result["diagnosis"]) > 0:
        diag_name = result["diagnosis"][0].get("name", "")
        api_diagnosis = get_icd11_diagnosis(diag_name)
        if api_diagnosis:
            result["diagnosis"] = [{"name": api_diagnosis["name"], "code": api_diagnosis["code"]}]
    
    return result

@app.route('/api/suggest', methods=['POST'])
def suggest():
    """
    API endpoint to process doctor's input and provide AI suggestions.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        completed_data = process_json_data(data)
        return jsonify(completed_data)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)