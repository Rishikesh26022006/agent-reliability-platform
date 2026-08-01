import sys
import json
import random
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_IN = PROJECT_ROOT / "data" / "training_examples.jsonl"
TRAIN_OUT = PROJECT_ROOT / "data" / "train.jsonl"
VAL_OUT = PROJECT_ROOT / "data" / "val.jsonl"

VAL_FRACTION = 0.2
SEED = 42

if __name__ == "__main__":
    examples = []
    with open(DATASET_IN) as f:
        for line in f:
            examples.append(json.loads(line))

    # Step 1: get unique trajectory IDs, and split THOSE (not the rows)
    trajectory_ids = sorted(set(e["trajectory_id"] for e in examples))
    random.seed(SEED)
    random.shuffle(trajectory_ids)

    n_val_trajs = max(1, int(len(trajectory_ids) * VAL_FRACTION))
    val_traj_ids = set(trajectory_ids[:n_val_trajs])
    train_traj_ids = set(trajectory_ids[n_val_trajs:])

    # Step 2: every row from a trajectory follows its trajectory's split assignment
    train_examples = [e for e in examples if e["trajectory_id"] in train_traj_ids]
    val_examples = [e for e in examples if e["trajectory_id"] in val_traj_ids]

    with open(TRAIN_OUT, "w") as f:
        for e in train_examples:
            f.write(json.dumps(e) + "\n")

    with open(VAL_OUT, "w") as f:
        for e in val_examples:
            f.write(json.dumps(e) + "\n")

    # Sanity check: confirm zero overlap
    overlap = train_traj_ids & val_traj_ids
    assert len(overlap) == 0, f"LEAK DETECTED: {overlap}"

    print(f"Train: {len(train_traj_ids)} trajectories, {len(train_examples)} rows")
    print(f"Val:   {len(val_traj_ids)} trajectories, {len(val_examples)} rows")
    print("No trajectory overlap between train/val - confirmed clean split")