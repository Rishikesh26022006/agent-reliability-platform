import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = PROJECT_ROOT / "data" / "trajectories"
LABELS_FILE = PROJECT_ROOT / "data" / "auto_labels.json"
DATASET_OUT = PROJECT_ROOT / "data" / "training_examples.jsonl"


def serialize_steps(steps: list) -> str:
    """Turn a list of trajectory steps into a single text block the model can read."""
    lines = []
    for step in steps:
        if step["type"] == "human":
            lines.append(f"CUSTOMER: {step['content']}")
        elif step["type"] == "tool_call":
            lines.append(f"AGENT_CALLS_TOOL: {step['tool']}({json.dumps(step['args'])})")
        elif step["type"] == "tool_result":
            lines.append(f"TOOL_RESULT: {step['content']}")
        elif step["type"] == "ai_response":
            lines.append(f"AGENT_REPLY: {step['content']}")
    return "\n".join(lines)


def build_examples_for_trajectory(traj_id: str, traj: dict, label: dict) -> list:
    """
    Create one training example per prefix of the trajectory.
    Label logic: only the FINAL prefix (the complete trajectory) gets the true
    success/failure label, since that's the only point we actually know the outcome.
    Earlier prefixes are labeled the same way for now (weak supervision) - the model
    learns to associate early warning patterns with eventual outcomes.
    """
    steps = traj["steps"]
    examples = []

    # Skip trajectories the auto-labeler couldn't parse or errored on
    if label.get("success") is None:
        return []

    is_failure = label["success"] is False
    category = label.get("failure_category") or "none"

    # Build a prefix at each meaningful checkpoint: after each tool_result or ai_response
    checkpoint_indices = [
        i for i, s in enumerate(steps) if s["type"] in ("tool_result", "ai_response")
    ]

    for idx in checkpoint_indices:
        prefix_steps = steps[: idx + 1]
        examples.append({
            "trajectory_id": traj_id,  # KEEP this - needed for correct train/val split later
            "prefix_length": idx + 1,
            "is_final_step": (idx == checkpoint_indices[-1]),
            "text": serialize_steps(prefix_steps),
            "label_is_failure": is_failure,
            "label_category": category,
        })

    return examples


if __name__ == "__main__":
    print("Script started")

    with open(LABELS_FILE) as f:
        labels = json.load(f)
    print(f"Loaded {len(labels)} label entries")

    all_examples = []
    skipped = 0

    for traj_id, label in labels.items():
        if not traj_id.startswith("traj_"):
            print(f"Skipping non-trajectory key: {traj_id}")
            skipped += 1
            continue

        traj_path = TRAJ_DIR / f"{traj_id}.json"
        if not traj_path.exists():
            print(f"Skipping {traj_id}: file not found at {traj_path}")
            skipped += 1
            continue

        with open(traj_path) as f:
            traj = json.load(f)

        if "steps" not in traj:
            print(f"WARNING: {traj_id} has no 'steps' key. Keys found: {list(traj.keys())}")
            skipped += 1
            continue

        examples = build_examples_for_trajectory(traj_id, traj, label)
        all_examples.extend(examples)

    with open(DATASET_OUT, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    n_trajectories = len(set(e["trajectory_id"] for e in all_examples))
    n_failure_trajs = len(set(
        e["trajectory_id"] for e in all_examples
        if e["is_final_step"] and e["label_is_failure"]
    ))

    print(f"\nBuilt {len(all_examples)} training examples from {n_trajectories} trajectories")
    print(f"({n_failure_trajs} of those trajectories are labeled as failures)")
    print(f"Skipped {skipped} labeled trajectories with missing files/keys")
    print(f"Saved to {DATASET_OUT}")