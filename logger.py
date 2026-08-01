import json
import time
import uuid
from pathlib import Path

LOG_DIR = Path("data/trajectories")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_trajectory(user_query: str, messages: list, model_name: str) -> str:
    """Convert a LangGraph message list into our structured trajectory format and save it."""
    trajectory_id = f"traj_{uuid.uuid4().hex[:8]}"
    steps = []

    for m in messages:
        if m.type == "human":
            steps.append({"type": "human", "content": m.content})
        elif m.type == "ai":
            if m.tool_calls:
                for tc in m.tool_calls:
                    steps.append({"type": "tool_call", "tool": tc["name"], "args": tc["args"]})
            elif m.content:
                text = m.content if isinstance(m.content, str) else " ".join(
                    block.get("text", "") for block in m.content if isinstance(block, dict)
                )
                steps.append({"type": "ai_response", "content": text})
        elif m.type == "tool":
            steps.append({"type": "tool_result", "content": m.content})

    trajectory = {
        "trajectory_id": trajectory_id,
        "query": user_query,
        "steps": steps,
        "model_used": model_name,
        "timestamp": time.time(),
        # These get filled in later during Week 2 labeling - empty for now
        "human_label": {"success": None, "failure_category": None, "notes": ""},
    }

    filepath = LOG_DIR / f"{trajectory_id}.json"
    with open(filepath, "w") as f:
        json.dump(trajectory, f, indent=2)

    return trajectory_id