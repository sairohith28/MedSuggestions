# Medical AI Assistant

This project is a Medical AI Assistant designed to assist healthcare professionals by providing structured outputs for diagnoses, investigations, medications, follow-up plans, and diet instructions based on patient data and doctor preferences. The assistant leverages a combination of MongoDB for data storage, a semantic model for similarity matching, and a Language Learning Model (LLM) API for generating suggestions.

## Features

- **Patient Data Management**: Store and retrieve patient visit records, medical history, personal history, family history, current medications, and allergies.
- **Doctor Preferences**: Fetch and utilize doctor preferences based on past consultations to provide personalized suggestions.
- **LLM Integration**: Use a Language Learning Model API to generate structured outputs for various medical fields.
- **Semantic Similarity**: Employ a semantic model to find similar past visits and doctor preferences.
- **ICD-11 Diagnosis**: Fetch ICD-11 diagnosis codes and names from an external API.
- **Few-Shot Learning**: Format examples for few-shot learning to improve LLM suggestions.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/medical-ai-assistant.git
    cd medical-ai-assistant
    ```

2. **Create a virtual environment and activate it**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    Create a `.env` file in the project root directory and add the following variables:
    ```env
    MONGO_URI=your_mongo_uri
    DB_NAME=your_database_name
    COLLECTION_NAME=your_collection_name
    LLM_API_URL=your_llm_api_url
    LLM_AUTH_HEADER=your_llm_auth_header
    ```

## Usage

1. **Run the Flask application**:
    ```bash
    python med5.py
    ```

2. **API Endpoint**:
    The main API endpoint `/api/suggest` accepts a POST request with JSON data containing `patientId` and `doctorId`. The endpoint processes the input and returns structured suggestions.

    Example request:
    ```json
    {
        "patientId": "60d5f9b8c2a1e2b8f8e8b8e8",
        "doctorId": "60d5f9b8c2a1e2b8f8e8b8e9"
    }
    ```

    Example response:
    ```json
    {
        "chief_complaints": [{"text": "High blood pressure and chest pain."}],
        "diagnosis": [{"code": "I10", "name": "Essential hypertension"}],
        "investigations": [{"code": "", "name": "ECG"}, {"code": "", "name": "Chest X-ray"}],
        "medications": [{"code": "", "name": "Amlodipine"}, {"code": "", "name": "Nitroglycerin"}],
        "followup": {"date": "2025-03-12", "text": "Review in 1 week with ECG results."},
        "diet_instructions": [{"text": "Reduce salt intake, avoid fatty foods, increase potassium-rich foods"}]
    }
    ```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](http://_vscodecontentref_/1) file for details.

## Contact

For any questions or suggestions, feel free to open an issue or contact the project maintainer at [vulapusairohith28@gmail.com](mailto:vulapusairohith28@gmail.com).

---

Thank you for using the Medical AI Assistant! We hope it helps you provide better care for your patients.
