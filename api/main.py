"""
FastAPI endpoint wrapping the ReliableAgentRunner.
POST /handle_ticket - run a support query through the reliability layer
GET  /trajectories  - list all logged trajectories with labels
GET  /metrics       - aggregate stats (total runs, correction rate, category breakdown)
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reliable_agent import ReliableAgentRunner

app = FastAPI(
    title="Agent Reliability Platform API",
    description="Customer support agent with real-time failure prediction and self-correction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

runner = ReliableAgentRunner(risk_threshold=0.35)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = PROJECT_ROOT / "data" / "trajectories"
LABELS_FILE = PROJECT_ROOT / "data" / "auto_labels.json"


class TicketRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class TicketResponse(BaseModel):
    trajectory_id: str
    query: str
    final_response: str
    risk_score: float
    risk_confidence: str
    correction_triggered: bool
    corrections: list
    latency_ms: Optional[float] = None


@app.post("/handle_ticket", response_model=TicketResponse)
def handle_ticket(req: TicketRequest):
    start = datetime.now()
    try:
        result = runner.run_with_guardrails(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency = (datetime.now() - start).total_seconds() * 1000

    # Extract clean text from final response
    raw_response = result["final_response"]
    if isinstance(raw_response, list):
        text = " ".join(
            part["text"] for part in raw_response
            if isinstance(part, dict) and "text" in part
        )
    else:
        text = str(raw_response)

    return TicketResponse(
        trajectory_id=result["trajectory_id"],
        query=req.query,
        final_response=text,
        risk_score=result["risk_info"]["risk_score"],
        risk_confidence=result["risk_info"]["confidence"],
        correction_triggered=result["risk_info"]["should_trigger_correction"],
        corrections=result["corrections_triggered"],
        latency_ms=round(latency, 1),
    )


@app.get("/trajectories")
def list_trajectories(limit: int = 50):
    labels = {}
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            labels = json.load(f)

    trajs = []
    for path in sorted(TRAJ_DIR.glob("traj_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        with open(path) as f:
            t = json.load(f)
        tid = t.get("trajectory_id", path.stem)
        label = labels.get(tid, {})
        trajs.append({
            "trajectory_id": tid,
            "query": t.get("query", ""),
            "timestamp": t.get("timestamp", ""),
            "model_used": t.get("model_used", ""),
            "step_count": len(t.get("steps", [])),
            "label_success": label.get("success"),
            "label_category": label.get("failure_category"),
        })
    return {"trajectories": trajs, "total": len(trajs)}


@app.get("/trajectories/{trajectory_id}")
def get_trajectory(trajectory_id: str):
    path = TRAJ_DIR / f"{trajectory_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Trajectory not found")
    with open(path) as f:
        traj = json.load(f)
    labels = {}
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            labels = json.load(f)
    traj["label"] = labels.get(trajectory_id, {})
    return traj


@app.get("/metrics")
def get_metrics():
    labels = {}
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            labels = json.load(f)

    traj_files = list(TRAJ_DIR.glob("traj_*.json"))
    total = len(traj_files)
    labeled = {k: v for k, v in labels.items() if k.startswith("traj_")}
    failures = [v for v in labeled.values() if v.get("success") is False]
    successes = [v for v in labeled.values() if v.get("success") is True]

    # Category breakdown
    categories: dict = {}
    for f in failures:
        cat = f.get("failure_category", "unknown") or "unknown"
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "total_trajectories": total,
        "labeled_trajectories": len(labeled),
        "failure_count": len(failures),
        "success_count": len(successes),
        "failure_rate": round(len(failures) / len(labeled), 3) if labeled else 0,
        "failure_categories": categories,
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
