import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from machine_service_predictor import (
    ServiceInput,
    load_model,
    predict_days_until_service,
    predict_next_service_date,
    explain_with_ollama,
    MODEL_PATH,
)

app = FastAPI(title="Machine Service Predictor API")
model = load_model(MODEL_PATH)

class PredictionRequest(BaseModel):
    machine_type: str
    issue_type: str
    severity: float
    parts_required: int
    technician_experience: float
    days_already_in_progress: int
    last_service_date: str | None = None

class PredictionResponse(BaseModel):
    predicted_days: int
    next_service_date: str
    explanation: str

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    last_service_date = None
    if req.last_service_date:
        last_service_date = datetime.datetime.strptime(req.last_service_date, "%Y-%m-%d").date()

    inp = ServiceInput(
        machine_type=req.machine_type,
        issue_type=req.issue_type,
        severity=req.severity,
        parts_required=req.parts_required,
        technician_experience=req.technician_experience,
        days_already_in_progress=req.days_already_in_progress,
        last_service_date=last_service_date,
    )

    days = predict_days_until_service(model, inp)
    next_date = predict_next_service_date(model, inp)

    prompt = (
        f"Given the machine with features: machine_type={inp.machine_type}, issue_type={inp.issue_type}, "
        f"severity={inp.severity}, parts_required={inp.parts_required}, technician_experience={inp.technician_experience}, "
        f"days_already_in_progress={inp.days_already_in_progress}, predicted_days_until_service={days}. "
        "Explain why this prediction was made and list suggested immediate actions and confidence indicators."
    )
    explanation = explain_with_ollama(prompt)

    return PredictionResponse(
        predicted_days=days,
        next_service_date=next_date.isoformat(),
        explanation=explanation,
    )
