import sys
import time
import random
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from taxonomy import format_taxonomy_for_prompt

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = PROJECT_ROOT / "data" / "trajectories"
LABELS_OUT = PROJECT_ROOT / "data" / "auto_labels.json"

# Use a capable model for labeling - accuracy matters more than cost here,
# but stick to a model with a generous free-tier quota
labeler_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

LABEL_PROMPT_TEMPLATE = """You are auditing an AI customer support agent's behavior for reliability issues.

Below is the full trajectory of one agent run: the customer's message, every tool call and result,
and the agent's final reply.

FAILURE TAXONOMY (only use these categories, or null if no failure):
{taxonomy}

TRAJECTORY:
{trajectory}

Evaluate this trajectory carefully. Check specifically:
1. Does the final reply match what the tools actually returned (correct order ID, amount, item)?
2. Did the agent verify any factual claims the customer made against tool data, when tool data was available?
3. Was the resolution (refund/escalate/deny) actually justified by the policy tool's output?
4. Is there anything hallucinated - stated as fact but not present in any tool result?

Respond ONLY with a JSON object, no other text, no code fences:
{{
  "success": true or false,
  "failure_category": "category_name_or_null",
  "confidence": "high" or "medium" or "low",
  "reasoning": "1-2 sentence explanation citing specific evidence from the trajectory"
}}
"""

# Trajectories you already manually flagged as interesting/failures - always include these
# NOTE: update these IDs to match your actual filenames in data/trajectories/
PRIORITY_IDS = [
    "traj_7a9501fe",  # unverified_claim (original)
    "traj_3769652e", "traj_7fb2f76b",  # inconsistent_policy_execution pair
    "traj_db3abbe8", "traj_59a6edf7",  # inconsistent_policy_execution pair 2
    "traj_3f4ad667", "traj_1565c315",  # premature_resolution_without_verification
    "traj_952a30b0",  # hallucinated_policy
    "traj_eb6128ad",  # hallucinated_policy (severe)
]


def extract_text(response) -> str:
    if isinstance(response.content, str):
        return response.content.strip()
    return " ".join(
        b.get("text", "") for b in response.content if isinstance(b, dict)
    ).strip()


def strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


def label_trajectory(traj: dict, max_retries: int = 4) -> dict:
    prompt = LABEL_PROMPT_TEMPLATE.format(
        taxonomy=format_taxonomy_for_prompt(),
        trajectory=json.dumps(traj["steps"], indent=2),
    )
    for attempt in range(max_retries):
        try:
            response = labeler_model.invoke(prompt)
            text = strip_code_fences(extract_text(response))
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {
                    "success": None,
                    "failure_category": "PARSE_ERROR",
                    "confidence": "low",
                    "reasoning": text[:200],
                }
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait = 15 * (attempt + 1)  # 15s, 30s, 45s, 60s
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries exceeded due to rate limiting")


if __name__ == "__main__":
    all_files = sorted(TRAJ_DIR.glob("*.json"))
    print(f"Found {len(all_files)} total trajectories")

    # Build the 25: priority ones first, then fill the rest randomly for variety
    priority_files = [f for f in all_files if f.stem in PRIORITY_IDS]
    remaining_files = [f for f in all_files if f.stem not in PRIORITY_IDS]
    random.seed(42)  # reproducible selection
    fill_count = max(0, 25 - len(priority_files))
    sampled_files = priority_files + random.sample(remaining_files, min(fill_count, len(remaining_files)))

    print(f"Selected {len(sampled_files)} trajectories to label (priority + random sample)")

    labels = {}
    if LABELS_OUT.exists():
        with open(LABELS_OUT) as f:
            labels = json.load(f)
        print(f"Resuming - {len(labels)} already labeled")

    for i, filepath in enumerate(sampled_files):
        traj_id = filepath.stem
        if traj_id in labels:
            continue  # skip already-labeled ones, so this script is safe to re-run

        with open(filepath) as f:
            traj = json.load(f)

        print(f"[{i+1}/{len(sampled_files)}] Labeling {traj_id}...")
        try:
            label = label_trajectory(traj)
            labels[traj_id] = label
            print(f"  -> success={label.get('success')}, category={label.get('failure_category')}")
        except Exception as e:
            print(f"  ERROR: {e}")
            labels[traj_id] = {"success": None, "failure_category": "ERROR", "reasoning": str(e)}

        # Save incrementally so a crash doesn't lose progress
        with open(LABELS_OUT, "w") as f:
            json.dump(labels, f, indent=2)

        time.sleep(4)  # stay under free-tier rate limits

    fail_count = sum(1 for l in labels.values() if l.get("success") is False)
    print(f"\nDone. {len(labels)} labeled. {fail_count} flagged as failures.")