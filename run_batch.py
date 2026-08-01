from agent_graph import run_agent
from logger import log_trajectory
import json
import time

with open("data/synthetic_queries.json") as f:
    TEST_QUERIES = json.load(f)

MAX_RETRIES = 2

def run_with_retry(query, max_retries=MAX_RETRIES):
    """Try running the agent, retrying on transient errors before giving up."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return run_agent(query), attempt  # attempt = how many retries it took
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt  # simple exponential backoff: 1s, 2s, 4s...
                print(f"  Attempt {attempt+1} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


if __name__ == "__main__":
    results = []
    for i, query in enumerate(TEST_QUERIES):
        print(f"\n=== Running query {i+1}/{len(TEST_QUERIES)} ===")
        print(query)
        try:
            messages, retries_used = run_with_retry(query)
            traj_id = log_trajectory(query, messages, model_name="gemini-3.5-flash-lite")
            final_reply = messages[-1].content
            if isinstance(final_reply, list):
                final_reply = " ".join(b.get("text", "") for b in final_reply if isinstance(b, dict))
            print(f"Logged as: {traj_id} (retries used: {retries_used})")
            print(f"Final reply: {final_reply}")
            results.append({"query": query, "traj_id": traj_id, "reply": final_reply, "retries_used": retries_used, "error": None})
        except Exception as e:
            print(f"FAILED after retries on query {i+1}: {e}")
            results.append({"query": query, "traj_id": None, "reply": None, "retries_used": MAX_RETRIES, "error": str(e)})
        time.sleep(1)

    with open("data/batch_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    success_count = sum(1 for r in results if r["error"] is None)
    retried_count = sum(1 for r in results if r.get("retries_used", 0) > 0)
    print(f"\n\nDone. {success_count}/{len(TEST_QUERIES)} ran without errors. {retried_count} needed at least one retry.")