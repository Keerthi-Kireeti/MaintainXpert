import joblib
import datetime
from dataclasses import dataclass
import pandas as pd
import json

MODEL_PATH = "model.pkl"

@dataclass
class ServiceInput:
    machine_type: str
    issue_type: str
    severity: float
    parts_required: int
    technician_experience: float
    days_already_in_progress: int
    last_service_date: datetime.date | None = None

def load_model(path):
    return joblib.load(path)

def predict_days_until_service(model, inp: ServiceInput):
    # Only use the features that the model was trained on
    features = {
        'severity': inp.severity,
        'parts_required': inp.parts_required,
        'technician_experience': inp.technician_experience,
        'days_already_in_progress': inp.days_already_in_progress
    }
    df = pd.DataFrame([features])
    return int(model.predict(df)[0])

def predict_next_service_date(model, inp: ServiceInput):
    days = predict_days_until_service(model, inp)
    start_date = inp.last_service_date or datetime.date.today()
    return start_date + datetime.timedelta(days=days)

def explain_with_ollama(prompt: str):
    try:
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": prompt}
        )
        # Handle streaming response from Ollama
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            if lines:
                # Get the last line which should contain the final response
                last_line = lines[-1]
                try:
                    data = json.loads(last_line)
                    return data.get("response", "No explanation available.")
                except json.JSONDecodeError:
                    return "No explanation available."
        return "No explanation available."
    except Exception as e:
        return f"No explanation available. Error: {str(e)}"
