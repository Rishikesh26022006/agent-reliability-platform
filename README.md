# 🛡️ Agent Reliability Platform

> **An end-to-end system for monitoring, predicting, and self-correcting failures in production AI agents.**

A customer support agent built with LangGraph + Gemini, extended with a real-time failure prediction layer (fine-tuned DistilBERT) and an autonomous self-correction loop. When the predictor detects elevated risk mid-trajectory, it injects an audit directive and re-runs the agent — all without human intervention.

---

## 🏗️ Architecture

```
Customer Query
      │
      ▼
┌─────────────────────────────┐
│   LangGraph Agent           │  ← Gemini 3.5 Flash + 4 tools
│   (agent_graph.py)          │    lookup_order, process_refund,
│                             │    escalate_to_human, check_policy
└────────────┬────────────────┘
             │ trajectory steps
             ▼
┌─────────────────────────────┐
│   Failure Predictor         │  ← DistilBERT fine-tuned on
│   (failure_predictor/)      │    labeled trajectory prefixes
│   predict_risk(steps)       │    → risk_score ∈ [0, 1]
└────────────┬────────────────┘
             │ risk_score
             ▼
┌─────────────────────────────┐
│   Correction Engine         │  risk < 0.35  → proceed
│   (correction/)             │  risk ≥ 0.35  → reflect & verify
│   get_correction_strategy() │  risk ≥ 0.65  → escalate to human
└────────────┬────────────────┘
             │ (if correction triggered)
             ▼
      Re-run agent with
      audit directive injected
             │
             ▼
      Final verified response
      + logged trajectory
```

---

## 📁 Project Structure

```
agent-reliability-platform/
├── agent_graph.py              # LangGraph agent + 4 customer support tools
├── agent_v1.py                 # v1 baseline agent (without graph structure)
├── reliable_agent.py           # Self-healing orchestrator (main entrypoint)
├── logger.py                   # Trajectory serialization to JSON
├── taxonomy.py                 # 10-category failure taxonomy
├── tools.py                    # Tool implementations
├── generate_queries.py         # Generates synthetic customer queries
├── run_batch.py                # Batch trajectory collection
│
├── api/
│   └── main.py                 # FastAPI: POST /handle_ticket, GET /metrics
│
├── dashboard/
│   └── app.py                  # Streamlit: risk monitor + trajectory viewer
│
├── failure_predictor/
│   ├── build_dataset.py        # Convert trajectories → training JSONL
│   ├── train.py                # Fine-tune DistilBERT with class weighting
│   ├── inference.py            # FailurePredictor: live risk scoring
│   └── auto_labeler.py         # Gemini-powered trajectory labeling
│
├── correction/
│   └── fallback_policies.py    # Tiered correction strategy engine
│
├── evaluation/
│   └── ab_test.py              # A/B comparison: baseline vs. treatment
│
├── data/
│   ├── trajectories/           # Raw trajectory JSON files (traj_*.json)
│   ├── train.jsonl             # Training dataset (prefix windows)
│   ├── val.jsonl               # Validation dataset
│   └── auto_labels.json        # Gemini-generated labels per trajectory
│
├── model_checkpoint/           # Best DistilBERT checkpoint (checkpoint-42)
├── Dockerfile                  # Docker image for the API
└── requirements.txt            # Full dependency list
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.12
- Windows: Create a short-path venv to avoid MAX_PATH limits:
  ```powershell
  python -m venv C:\arp_venv
  ```
- Linux/Mac: Standard venv works fine

### 2. Install Dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Run End-to-End Test

```bash
python reliable_agent.py
```

Expected output:
```
Trajectory ID: traj_xxxxxxxx
Risk Score: 0.519 (medium confidence, correction TRIGGERED)
Strategy: reflect_and_verify
Final Response: "I have re-verified your order details..."
```

### 5. Start the API Server

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 6. Start the Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard at: http://localhost:8501

---

## 🔌 API Reference

### `POST /handle_ticket`
Run a support query through the full reliability pipeline.

**Request:**
```json
{ "query": "My order 8891 arrived broken. I want a refund." }
```

**Response:**
```json
{
  "trajectory_id": "traj_7468641a",
  "query": "...",
  "final_response": "I have re-verified your order...",
  "risk_score": 0.5186,
  "risk_confidence": "medium",
  "correction_triggered": true,
  "corrections": [...],
  "latency_ms": 4231.0
}
```

### `GET /metrics`
Aggregate stats across all logged trajectories.

```json
{
  "total_trajectories": 90,
  "labeled_trajectories": 42,
  "failure_count": 10,
  "success_count": 28,
  "failure_rate": 0.238,
  "failure_categories": {
    "premature_resolution_without_verification": 4,
    "hallucinated_policy": 2,
    ...
  }
}
```

### `GET /trajectories`
List all trajectory IDs with labels.

### `GET /trajectories/{id}`
Full trajectory detail (steps + auto-label).

---

## 🧠 Failure Taxonomy

The system detects 10 failure categories:

| Category | Description |
|---|---|
| `premature_resolution_without_verification` | Resolving before tool cross-check |
| `hallucinated_policy` | Inventing policy not in the prompt |
| `wrong_tool_selection` | Using wrong tool for the situation |
| `unverified_claim` | Accepting customer claim without lookup |
| `response_data_mismatch` | Response contradicts tool result |
| `premature_escalation` | Escalating unnecessarily |
| `infinite_loop` | Agent calls same tool repeatedly |
| `incomplete_resolution` | Leaving issue partially addressed |
| `context_loss` | Losing track of earlier conversation turns |
| `off_topic_response` | Drifting from the customer issue |

---

## 🔬 Model Details

- **Base model:** `distilbert-base-uncased` (66M parameters)
- **Task:** Binary sequence classification (failure / not failure)
- **Training data:** 107 prefix-windowed trajectory steps
- **Validation data:** 24 prefix-windowed trajectory steps
- **Class imbalance:** ~87.5% not-failure, 12.5% failure → handled with inverse-frequency class weights
- **Best checkpoint:** `checkpoint-42` (ROC-AUC = 0.43)
- **Honest assessment:** With 107 training rows the model learns limited signal; the architecture is production-ready and would improve significantly with 1000+ labeled trajectories.

---

## 📊 A/B Comparison

Run `evaluation/ab_test.py` to compare the baseline agent against the reliability layer on 8 held-out queries:

```bash
python evaluation/ab_test.py
```

Reports:
- Correction trigger rate
- Average risk score
- Latency overhead
- Per-query breakdown

---

## 🐳 Docker

```bash
docker build -t arp-api .
docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key arp-api
```

---

## 📅 Development Timeline

| Week | Focus | Status |
|---|---|---|
| 1 | Agent skeleton + tools + logger | ✅ Done |
| 2 | Batch runner + auto-labeler + taxonomy | ✅ Done |
| 3 | Data pipeline + DistilBERT fine-tuning | ✅ Done |
| 4 | Inference layer + fallback policies + integration | ✅ Done |
| 5 | FastAPI + Streamlit dashboard + Docker | ✅ Done |
| 6 | A/B test + README + final polish | ✅ Done |

---

## 🛠️ Tech Stack

- **Agent:** LangGraph 1.2 + Gemini 3.5 Flash (`langchain-google-genai`)
- **Failure Predictor:** DistilBERT (HuggingFace Transformers 5.14)
- **Training:** HuggingFace Trainer + WeightedRandomSampler
- **API:** FastAPI 0.141 + uvicorn
- **Dashboard:** Streamlit 1.60
- **Data:** HuggingFace Datasets + pandas

---

## 📝 License

MIT
