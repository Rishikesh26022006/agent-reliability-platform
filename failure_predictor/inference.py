import json
import os
from pathlib import Path

# Optional heavy ML import with fallback for memory-constrained cloud environments (e.g. Render 512MB RAM)
HAS_TORCH = False
try:
    if os.environ.get("USE_LIGHTWEIGHT_PREDICTOR", "false").lower() != "true":
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        HAS_TORCH = True
except Exception as e:
    print(f"PyTorch/Transformers not available or disabled: {e}. Using Lightweight Failure Predictor.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "failure_predictor" / "model_checkpoint" / "checkpoints" / "checkpoint-42"
FALLBACK_MODEL_DIR = PROJECT_ROOT / "failure_predictor" / "model_checkpoint"

class FailurePredictor:
    def __init__(self, model_dir=None, threshold: float = 0.35):
        self.threshold = threshold
        self.use_heavy_model = HAS_TORCH

        if self.use_heavy_model:
            if model_dir is None:
                if DEFAULT_MODEL_DIR.exists():
                    model_dir = DEFAULT_MODEL_DIR
                else:
                    model_dir = FALLBACK_MODEL_DIR

            self.model_dir = Path(model_dir)
            base_model_name = "distilbert-base-uncased"
            load_path = str(self.model_dir) if self.model_dir.exists() else base_model_name

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(load_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
                self.model.eval()
            except Exception as e:
                print(f"Warning: Could not load heavy model ({e}). Falling back to Lightweight Predictor.")
                self.use_heavy_model = False

    def serialize_steps(self, steps: list) -> str:
        lines = []
        for step in steps:
            stype = step.get("type")
            if stype == "human":
                lines.append(f"CUSTOMER: {step.get('content', '')}")
            elif stype == "tool_call":
                args_str = json.dumps(step.get("args", {}))
                lines.append(f"AGENT_CALLS_TOOL: {step.get('tool', '')}({args_str})")
            elif stype == "tool_result":
                lines.append(f"TOOL_RESULT: {step.get('content', '')}")
            elif stype == "ai_response":
                lines.append(f"AGENT_REPLY: {step.get('content', '')}")
        return "\n".join(lines)

    def _heuristic_risk(self, steps: list) -> dict:
        """Lightweight zero-memory risk evaluator for cloud environments."""
        text = self.serialize_steps(steps)
        if not text.strip():
            return {"risk_score": 0.0, "should_trigger_correction": False, "confidence": "low", "text_snippet": ""}

        score = 0.20  # Base risk

        # Rule 1: Customer requesting refund without order lookup tool call
        has_order_lookup = any(s.get("type") == "tool_call" and s.get("tool") == "lookup_order" for s in steps)
        refund_requested = any("refund" in str(s.get("content", "")).lower() for s in steps if s.get("type") == "human")
        if refund_requested and not has_order_lookup:
            score += 0.35

        # Rule 2: Repeated tool calls (infinite loop risk)
        tool_calls = [s.get("tool") for s in steps if s.get("type") == "tool_call"]
        if len(tool_calls) != len(set(tool_calls)):
            score += 0.25

        # Rule 3: Processed refund without policy check
        has_refund_call = any(s.get("type") == "tool_call" and s.get("tool") == "process_refund" for s in steps)
        has_policy_check = any(s.get("type") == "tool_call" and s.get("tool") == "check_policy" for s in steps)
        if has_refund_call and not has_policy_check:
            score += 0.20

        score = min(max(round(score, 4), 0.0), 1.0)
        should_trigger = score >= self.threshold
        conf = "high" if score >= 0.60 else ("medium" if score >= self.threshold else "low")

        return {
            "risk_score": score,
            "should_trigger_correction": should_trigger,
            "confidence": conf,
            "text_snippet": text[-300:],
        }

    def predict_risk(self, steps: list) -> dict:
        if not self.use_heavy_model:
            return self._heuristic_risk(steps)

        text = self.serialize_steps(steps)
        if not text.strip():
            return {"risk_score": 0.0, "should_trigger_correction": False, "confidence": "low", "text_snippet": ""}

        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            failure_prob = probs[0][1].item()

        risk_score = round(failure_prob, 4)
        should_trigger = risk_score >= self.threshold
        conf = "high" if risk_score >= 0.60 else ("medium" if risk_score >= self.threshold else "low")

        return {
            "risk_score": risk_score,
            "should_trigger_correction": should_trigger,
            "confidence": conf,
            "text_snippet": text[-300:],
        }
