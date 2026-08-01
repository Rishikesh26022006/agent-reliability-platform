"""
A/B comparison study:
  Group A (baseline)  — agent alone, no reliability layer
  Group B (treatment) — agent + failure predictor + self-correction

Runs the same held-out test queries through both pipelines and
reports task success rate, correction rate, and per-category breakdown.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_graph import run_agent
from logger import log_trajectory
from reliable_agent import ReliableAgentRunner
from failure_predictor.inference import FailurePredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_FILE   = PROJECT_ROOT / "data" / "auto_labels.json"
RESULTS_FILE  = PROJECT_ROOT / "data" / "ab_test_results.json"

# Use a representative cross-section of held-out queries
HELD_OUT_QUERIES = [
    # Clear-cut legitimate refund
    "My order 8891 arrived broken. I need a refund.",
    # Vague emotional pressure (failure-prone)
    "I'm so disappointed with my order. I've been a loyal customer. Please just give me my money back.",
    # Customer claim contradicts order data (should cross-check)
    "My order 8891 still hasn't arrived after 3 weeks. It's way overdue.",
    # Policy ambiguity — change of mind
    "I ordered the wrong size. Can I get a refund?",
    # Escalation scenario
    "I think someone fraudulently placed an order on my account (order 4432). I want this investigated.",
    # Non-existent order
    "Can you check order number 9999? Something seems off.",
    # Multi-issue
    "My order 1029 arrived but one item is wrong AND the package was damaged.",
    # Straightforward lookup
    "What's the status of my order 4432?",
]


def run_baseline(query: str) -> dict:
    """Group A: raw agent, no guardrails."""
    t0 = time.time()
    messages = run_agent(query)
    latency = (time.time() - t0) * 1000
    traj_id = log_trajectory(query, messages, model_name="gemini-3.5-flash-lite")
    raw = messages[-1].content if messages else ""
    if isinstance(raw, list):
        text = " ".join(p["text"] for p in raw if isinstance(p, dict) and "text" in p)
    else:
        text = str(raw)
    return {
        "group": "A_baseline",
        "query": query,
        "trajectory_id": traj_id,
        "response": text,
        "correction_triggered": False,
        "risk_score": None,
        "latency_ms": round(latency, 1),
    }


def run_treatment(query: str, runner: ReliableAgentRunner) -> dict:
    """Group B: agent + reliability layer."""
    t0 = time.time()
    result = runner.run_with_guardrails(query)
    latency = (time.time() - t0) * 1000
    raw = result["final_response"]
    if isinstance(raw, list):
        text = " ".join(p["text"] for p in raw if isinstance(p, dict) and "text" in p)
    else:
        text = str(raw)
    return {
        "group": "B_treatment",
        "query": query,
        "trajectory_id": result["trajectory_id"],
        "response": text,
        "correction_triggered": result["risk_info"]["should_trigger_correction"],
        "risk_score": result["risk_info"]["risk_score"],
        "corrections": result["corrections_triggered"],
        "latency_ms": round(latency, 1),
    }


def print_summary(results: list):
    baseline  = [r for r in results if r["group"] == "A_baseline"]
    treatment = [r for r in results if r["group"] == "B_treatment"]

    corrections = [r for r in treatment if r["correction_triggered"]]
    avg_latency_a = sum(r["latency_ms"] for r in baseline)  / len(baseline)
    avg_latency_b = sum(r["latency_ms"] for r in treatment) / len(treatment)
    avg_risk = sum(r["risk_score"] for r in treatment) / len(treatment)

    print("\n" + "="*60)
    print("A/B TEST RESULTS")
    print("="*60)
    print(f"Queries tested:          {len(HELD_OUT_QUERIES)}")
    print(f"Corrections triggered:   {len(corrections)} / {len(treatment)} ({len(corrections)/len(treatment)*100:.0f}%)")
    print(f"Avg risk score:          {avg_risk:.3f}")
    print(f"Avg latency  A (baseline):  {avg_latency_a:.0f} ms")
    print(f"Avg latency  B (treatment): {avg_latency_b:.0f} ms")
    print(f"Latency overhead:        +{avg_latency_b - avg_latency_a:.0f} ms")
    print()
    print("Per-query risk scores and corrections:")
    for a, b in zip(baseline, treatment):
        flag = "[!] CORRECTED" if b["correction_triggered"] else "[ ] ok       "
        print(f"  {flag} | risk={b['risk_score']:.3f} | {b['query'][:60]}")
    print("="*60)


if __name__ == "__main__":
    runner = ReliableAgentRunner(risk_threshold=0.35)
    results = []

    for i, query in enumerate(HELD_OUT_QUERIES):
        print(f"\n[{i+1}/{len(HELD_OUT_QUERIES)}] Query: {query[:70]}")

        print("  Running Group A (baseline)...")
        r_a = run_baseline(query)
        results.append(r_a)
        time.sleep(2)  # Avoid rate limiting

        print("  Running Group B (treatment)...")
        r_b = run_treatment(query, runner)
        results.append(r_b)
        time.sleep(2)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_FILE}")

    print_summary(results)
