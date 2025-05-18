from dotenv import load_dotenv
load_dotenv()

import json
import requests
import logging
import re
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, TypedDict, Optional, Union, Literal, cast
from flask import Flask, request, jsonify
from pytz import timezone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Timezone configuration
IST = timezone("Asia/Kolkata")

# Create Flask app
app = Flask(__name__)

# Custom LLM Client for extraction
class CustomExtractClient:
    def __init__(self):
        self.api_endpoint = os.getenv("14B_URL")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer apex@#1"
        }
        self.max_retries = 3
        
    def extract(self, field: str, prompt: str, examples: str = "") -> str:
        """Extract information from clinical notes using LLM"""
        system_prompt = f"You are an expert medical assistant that extracts {field} from clinical notes. Return only the extracted information in the requested format."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{examples}\n\n{prompt}"}
        ]
        
        payload = {
            "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.1
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending request to extract {field}")
                response = requests.post(
                    self.api_endpoint, 
                    json=payload,
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    logger.info(f"Received response for {field}")
                    result = response.json()
                    if "choices" in result and result["choices"]:
                        content = result["choices"][0].get("message", {}).get("content", "")
                        if content:
                            logger.info(f"Successfully extracted {field}")
                            return content.strip()
                else:
                    logger.error(f"Error response from API: {response.status_code}")
                    logger.error(response.text)
            
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            
            # Wait before retrying
            time.sleep(2)
        
        logger.warning(f"Failed to extract {field} after {self.max_retries} attempts")
        return ""

# Clinical Note Extractor
class ClinicalNoteExtractor:
    def __init__(self):
        self.llm_client = CustomExtractClient()
    
    def extract_clinical_note(self, clinical_note: str) -> Dict[str, Any]:
        """Extract all relevant sections from a clinical note"""
        logger.info("Starting extraction of clinical note")
        
        # Initialize the result structure
        result = self._get_empty_template()
        
        # Extract the sections in parallel
        result["vitals"] = self._extract_vitals(clinical_note)
        result["chief_complaints"] = self._extract_chief_complaints(clinical_note)
        result["current_medications"] = self._extract_current_medications(clinical_note)
        result["allergies"] = self._extract_allergies(clinical_note)
        result["medical_history"] = self._extract_medical_history(clinical_note)
        result["family_history"] = self._extract_family_history(clinical_note)
        result["personal_history"] = self._extract_personal_history(clinical_note)
        result["investigations"] = self._extract_investigations(clinical_note)
        result["diagnosis"] = self._extract_diagnosis(clinical_note)
        result["medications"] = self._extract_medications(clinical_note)
        result["diet_instructions"] = self._extract_diet_instructions(clinical_note)
        result["followup"] = self._extract_followup(clinical_note)
        
        logger.info("Completed extraction of clinical note")
        return result
    
    def _extract_vitals(self, text: str) -> Dict[str, Any]:
        """Extract vital signs from clinical notes"""
        prompt = f"""Extract the following vital signs from the clinical note:
- Blood pressure (systolic and diastolic)
- Respiratory rate
- Pulse/heart rate
- Oxygen saturation (SpO2)
- Temperature
- Weight
- Height
- BMI

Return a JSON object with the following structure:
{{
    "bp": {{
        "systolic": string,
        "diastolic": string
    }},
    "resp": string,
    "pulse": string,
    "spo2": string,
    "temperature": string,
    "weight": string,
    "height": string,
    "bmi": string
}}

Clinical note:
{text}
"""
        try:
            response = self.llm_client.extract("vital signs", prompt)
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    vitals = json.loads(json_match.group(0))
                    # Ensure the correct structure
                    if isinstance(vitals, dict):
                        if "bp" not in vitals or not isinstance(vitals["bp"], dict):
                            vitals["bp"] = {"systolic": "", "diastolic": ""}
                        
                        # Make sure all expected fields exist
                        for field in ["resp", "pulse", "spo2", "temperature", "weight", "height", "bmi"]:
                            if field not in vitals:
                                vitals[field] = ""
                        
                        return vitals
                except json.JSONDecodeError:
                    pass
            
            # Fallback extraction using regex
            bp_sys = re.search(r'(?:BP|Blood Pressure).*?(\d+)[/]', text)
            bp_dia = re.search(r'(?:BP|Blood Pressure).*?[/](\d+)', text)
            pulse = re.search(r'(?:Pulse|HR).*?(\d+)', text)
            resp = re.search(r'(?:Resp|Respiratory Rate).*?(\d+)', text)
            spo2 = re.search(r'(?:SpO2|Oxygen Saturation).*?(\d+)', text)
            temp = re.search(r'(?:Temp|Temperature).*?([\d\.]+)', text)
            weight = re.search(r'(?:Weight).*?([\d\.]+)', text)
            height = re.search(r'(?:Height).*?([\d\.]+)', text)
            bmi = re.search(r'(?:BMI).*?([\d\.]+)', text)
            
            return {
                "bp": {
                    "systolic": bp_sys.group(1) if bp_sys else "",
                    "diastolic": bp_dia.group(1) if bp_dia else ""
                },
                "resp": resp.group(1) if resp else "",
                "pulse": pulse.group(1) if pulse else "",
                "spo2": spo2.group(1) if spo2 else "",
                "temperature": temp.group(1) if temp else "",
                "weight": weight.group(1) if weight else "",
                "height": height.group(1) if height else "",
                "bmi": bmi.group(1) if bmi else ""
            }
            
        except Exception as e:
            logger.error(f"Error extracting vitals: {str(e)}")
            return {
                "bp": {"systolic": "", "diastolic": ""},
                "resp": "",
                "pulse": "",
                "spo2": "",
                "temperature": "",
                "weight": "",
                "height": "",
                "bmi": ""
            }
    
    def _extract_chief_complaints(self, text: str) -> str:
        """Extract chief complaints from clinical notes"""
        prompt = f"""Extract the chief complaints from the clinical note. 
The chief complaint is the primary reason why the patient sought medical care.
Return ONLY the chief complaints as a simple string.

Clinical note:
{text}
"""
        try:
            response = self.llm_client.extract("chief complaints", prompt)
            return response.strip('"\'{}[]')
        except Exception as e:
            logger.error(f"Error extracting chief complaints: {str(e)}")
            return ""
    
    def _extract_current_medications(self, text: str) -> str:
        """Extract current medications from clinical notes"""
        prompt = f"""Extract the current medications the patient is taking from the clinical note.
Current medications are medications the patient was taking BEFORE this visit, not new prescriptions.
Return ONLY the list of current medications as a simple string.

Clinical note:
{text}
"""
        try:
            response = self.llm_client.extract("current medications", prompt)
            return response.strip('"\'{}[]')
        except Exception as e:
            logger.error(f"Error extracting current medications: {str(e)}")
            return ""
    
    def _extract_allergies(self, text: str) -> str:
        """Extract allergies from clinical notes"""
        prompt = f"""Extract any allergies mentioned in the clinical note.
Include the type of reaction if mentioned.
Return ONLY the allergies as a simple string.

Clinical note:
{text}
"""
        try:
            response = self.llm_client.extract("allergies", prompt)
            return response.strip('"\'{}[]')
        except Exception as e:
            logger.error(f"Error extracting allergies: {str(e)}")
            return ""
    
    def _extract_medical_history(self, text: str) -> str:
        """Extract medical history from clinical notes"""
        prompt = f"""Extract the patient's past medical history from the clinical note.
This includes chronic conditions, past surgeries, and major illnesses.
Return ONLY the medical history as a simple string.

Clinical note:
{text}
"""
        try:
            response = self.llm_client.extract("medical history", prompt)
            return response.strip('"\'{}[]')
        except Exception as e:
            logger.error(f"Error extracting medical history: {str(e)}")
            return ""
    
    def _extract_family_history(self, text: str) -> str:
        """Extract family history from clinical notes"""
        prompt = f"""Extract the patient's family history from the clinical note.
This includes medical conditions affecting blood relatives.
Return ONLY the family history as a simple string.

Clinical note:
{text}
"""
        try:
            response = self.llm_client.extract("family history", prompt)
            return response.strip('"\'{}[]')
        except Exception as e:
            logger.error(f"Error extracting family history: {str(e)}")
            return ""
    
    def _extract_personal_history(self, text: str) -> str:
        """Extract personal history from clinical notes"""
        prompt = f"""Extract the patient's personal history from the clinical note.
This includes smoking status, alcohol use, occupation, exercise habits, etc.
Return ONLY the personal history as a simple string.

Clinical note:
{text}
"""
        try:
            response = self.llm_client.extract("personal history", prompt)
            return response.strip('"\'{}[]')
        except Exception as e:
            logger.error(f"Error extracting personal history: {str(e)}")
            return ""
    
    def _extract_investigations(self, text: str) -> List[Dict[str, str]]:
        """Extract investigations from clinical notes"""
        prompt = f"""Extract all investigations (labs, imaging, etc.) ordered or performed during this visit.
Return a JSON array of objects with a 'name' field. For example:
[
    {{"name": "CBC"}},
    {{"name": "Chest X-ray"}}
]

Clinical note:
{text}
"""
        try:
            response = self.llm_client.extract("investigations", prompt)
            
            # Try to extract JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, list):
                        return [{"code": "", "name": item.get("name", "")} for item in parsed if isinstance(item, dict)]
                except:
                    pass
            
            # Fallback extraction
            investigations = []
            # Look for common investigations in the text
            for investigation in ["CBC", "X-ray", "CT", "MRI", "Ultrasound", "EKG", "ECG", "Blood test", "Urinalysis"]:
                if re.search(rf'\b{re.escape(investigation)}\b', text, re.IGNORECASE):
                    investigations.append({"code": "", "name": investigation})
            
            return investigations if investigations else [{"code": "", "name": ""}]
            
        except Exception as e:
            logger.error(f"Error extracting investigations: {str(e)}")
            return [{"code": "", "name": ""}]
    
    def _extract_diagnosis(self, text: str) -> List[Dict[str, str]]:
        """Extract diagnoses from clinical notes"""
        prompt = f"""Extract all diagnoses made during this visit.
Look for the Assessment or Diagnosis section.
Return a JSON array of objects with a 'name' field. For example:
[
    {{"name": "Hypertension"}},
    {{"name": "Type 2 Diabetes"}}
]

Clinical note:
{text}
"""
        try:
            response = self.llm_client.extract("diagnoses", prompt)
            
            # Try to extract JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, list):
                        return [{"code": "", "name": item.get("name", "")} for item in parsed if isinstance(item, dict)]
                except:
                    pass
            
            # Fallback extraction - look for the Assessment section
            assessment_match = re.search(r'(?:ASSESSMENT|IMPRESSION|DIAGNOSIS)[:\s]*(.*?)(?:PLAN|TREATMENT|$)', text, re.DOTALL | re.IGNORECASE)
            if assessment_match:
                diagnoses = []
                diagnosis_lines = assessment_match.group(1).strip().split('\n')
                
                for line in diagnosis_lines:
                    # Remove numbering and clean up
                    clean_line = re.sub(r'^\d+\.\s*', '', line).strip()
                    if clean_line:
                        diagnoses.append({"code": "", "name": clean_line})
                
                return diagnoses if diagnoses else [{"code": "", "name": ""}]
            
            return [{"code": "", "name": ""}]
            
        except Exception as e:
            logger.error(f"Error extracting diagnoses: {str(e)}")
            return [{"code": "", "name": ""}]
    
    def _extract_medications(self, text: str) -> List[Dict[str, str]]:
        """Extract prescribed medications from clinical notes"""
        prompt = f"""Extract all medications prescribed during THIS visit (not current medications).
Look for the Plan or Medications section.
Return a JSON array of objects with 'name' and 'instructions' fields. For example:
[
    {{
        "name": "Amoxicillin 500mg",
        "instructions": "Take one capsule three times a day for 10 days"
    }}
]

Clinical note:
{text}
"""
        try:
            response = self.llm_client.extract("medications", prompt)
            
            # Try to extract JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, list):
                        return [{"code": "", "name": item.get("name", ""), "instructions": item.get("instructions", "")} 
                                for item in parsed if isinstance(item, dict)]
                except:
                    pass
            
            # Fallback extraction - look for medications in the Plan section
            plan_match = re.search(r'(?:PLAN|TREATMENT|PRESCRIBED)[:\s]*(.*?)(?:FOLLOW|DIET|INSTRUCTIONS|$)', text, re.DOTALL | re.IGNORECASE)
            if plan_match:
                medications = []
                plan_text = plan_match.group(1).strip()
                
                # Look for medication patterns - drug name followed by dosage and/or instructions
                med_matches = re.finditer(r'([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(\d+(?:\.\d+)?(?:mg|g|mcg|ml))\s*([^\.]*?)(?:\.|$)', plan_text)
                
                for match in med_matches:
                    medications.append({
                        "code": "",
                        "name": f"{match.group(1)} {match.group(2)}",
                        "instructions": match.group(3).strip()
                    })
                
                return medications if medications else [{"code": "", "name": "", "instructions": ""}]
            
            return [{"code": "", "name": "", "instructions": ""}]
            
        except Exception as e:
            logger.error(f"Error extracting medications: {str(e)}")
            return [{"code": "", "name": "", "instructions": ""}]
    
    def _extract_diet_instructions(self, text: str) -> str:
        """Extract diet instructions from clinical notes"""
        prompt = f"""Extract any dietary recommendations or instructions from the clinical note.
Look for the Diet Instructions section.
Return ONLY the diet instructions as a simple string.

Clinical note:
{text}
"""
        try:
            response = self.llm_client.extract("diet instructions", prompt)
            return response.strip('"\'{}[]')
        except Exception as e:
            logger.error(f"Error extracting diet instructions: {str(e)}")
            return ""
    
    def _extract_followup(self, text: str) -> Dict[str, str]:
        """Extract followup instructions from clinical notes"""
        today = datetime.now(IST).date()
        prompt = f"""Extract the follow-up information from the clinical note.
Look for the follow-up appointment date and any related instructions.
Return a JSON object with 'text' and 'date' fields. The date should be in YYYY-MM-DD format.
Today's date is: {today.strftime('%Y-%m-%d')}

For example:
{{
    "text": "Return for follow-up in 2 weeks",
    "date": "2025-05-28"
}}

Clinical note:
{text}
"""
        try:
            response = self.llm_client.extract("followup", prompt)
            
            # Try to extract JSON object
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, dict):
                        return {
                            "text": parsed.get("text", ""),
                            "date": parsed.get("date", "")
                        }
                except:
                    pass
            
            # Fallback extraction - try to find follow-up information
            followup_text = ""
            followup_date = ""
            
            # Look for follow-up text
            followup_match = re.search(r'(?:follow\s*up|return|come\s*back).*?(?:in\s+(\d+)\s+(day|week|month)s?)', text, re.IGNORECASE)
            if followup_match:
                followup_text = f"Follow up in {followup_match.group(1)} {followup_match.group(2)}(s)"
                
                # Calculate the date
                num = int(followup_match.group(1))
                unit = followup_match.group(2).lower()
                
                if unit == 'day':
                    followup_date = (today + timedelta(days=num)).strftime('%Y-%m-%d')
                elif unit == 'week':
                    followup_date = (today + timedelta(weeks=num)).strftime('%Y-%m-%d')
                elif unit == 'month':
                    # Approximate months as 30 days
                    followup_date = (today + timedelta(days=num*30)).strftime('%Y-%m-%d')
            
            # Check for explicit date
            date_match = re.search(r'(?:followup|return|come\s*back|appointment).*?(?:on|at)?\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', text, re.IGNORECASE)
            if date_match:
                day = int(date_match.group(1))
                month = int(date_match.group(2))
                year = int(date_match.group(3))
                if year < 100:  # Two-digit year
                    year += 2000
                
                try:
                    followup_date = datetime(year, month, day).strftime('%Y-%m-%d')
                except ValueError:
                    pass
            
            return {
                "text": followup_text,
                "date": followup_date
            }
            
        except Exception as e:
            logger.error(f"Error extracting followup: {str(e)}")
            return {"text": "", "date": ""}
    
    def _get_empty_template(self) -> Dict[str, Any]:
        """Returns an empty template for extraction results"""
        return {
            "vitals": {
                "bp": {"systolic": "", "diastolic": ""},
                "resp": "",
                "pulse": "",
                "spo2": "",
                "temperature": "",
                "weight": "",
                "height": "",
                "bmi": ""
            },
            "chief_complaints": "",
            "current_medications": "",
            "allergies": "",
            "medical_history": "",
            "family_history": "",
            "personal_history": "",
            "investigations": [{"code": "", "name": ""}],
            "diagnosis": [{"code": "", "name": ""}],
            "medications": [{"code": "", "name": "", "instructions": ""}],
            "diet_instructions": "",
            "followup": {"text": "", "date": ""}
        }

# Create an instance of the extractor
extractor = ClinicalNoteExtractor()

@app.route("/extract", methods=["POST"])
def extract_note():
    try:
        data = request.get_json()
        if not data or "input_text" not in data:
            return jsonify({
                "status": False,
                "data": {},
                "message": "Missing input_text field"
            }), 400

        result = extractor.extract_clinical_note(data["input_text"])
        return jsonify({
            "status": True,
            "data": result,
            "message": "Successfully extracted data from clinical note"
        })

    except Exception as e:
        logger.error(f"Error in extract_note endpoint: {str(e)}")
        return jsonify({
            "status": False,
            "data": {},
            "message": f"Error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)