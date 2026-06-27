import json
import logging
from pathlib import Path

import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Load model at startup ─────────────────────────────────────────────────────
MODEL_PATH = Path("data/model/best_model")
model = None
metadata = {}

def load_model():
    global model, metadata
    if not MODEL_PATH.exists():
        logger.warning("No trained model found. Run train.py first.")
        return
    model = mlflow.sklearn.load_model(str(MODEL_PATH))
    meta_path = Path("data/model/metadata.json")
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    logger.info(f"Model loaded: {metadata.get('best_model', 'unknown')}")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Support Ticket Classifier",
    description="Classifies support tickets into billing, technical, general, or urgent.",
    version="1.0.0",
)

@app.on_event("startup")
def startup_event():
    load_model()

# ── Request / Response schemas ────────────────────────────────────────────────
class TicketRequest(BaseModel):
    text: str
    ticket_id: str = "unknown"

class TicketResponse(BaseModel):
    ticket_id: str
    text: str
    predicted_category: str
    model_used: str
    model_accuracy: float

class BatchRequest(BaseModel):
    tickets: list[TicketRequest]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "model": metadata.get("best_model", "none"),
        "accuracy": metadata.get("accuracy", 0),
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}

@app.post("/classify", response_model=TicketResponse)
def classify(request: TicketRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    prediction = model.predict([request.text])[0]

    return TicketResponse(
        ticket_id=request.ticket_id,
        text=request.text,
        predicted_category=prediction,
        model_used=metadata.get("best_model", "unknown"),
        model_accuracy=round(metadata.get("accuracy", 0), 3),
    )

@app.post("/classify/batch")
def classify_batch(request: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")
    if not request.tickets:
        raise HTTPException(status_code=400, detail="No tickets provided.")

    texts = [t.text for t in request.tickets]
    predictions = model.predict(texts)

    results = []
    for ticket, prediction in zip(request.tickets, predictions):
        results.append({
            "ticket_id": ticket.ticket_id,
            "text": ticket.text,
            "predicted_category": prediction,
            "model_used": metadata.get("best_model", "unknown"),
        })

    return {"total": len(results), "results": results}
