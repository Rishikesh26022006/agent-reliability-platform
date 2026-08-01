import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "failure_predictor" / "model_checkpoint" / "checkpoints" / "checkpoint-42"
FALLBACK_MODEL_DIR = PROJECT_ROOT / "failure_predictor" / "model_checkpoint"

class FailurePredictor:
    def __init__(self, model_dir=None, threshold: float = 0.35):
        if model_dir is None:
            if DEFAULT_MODEL_DIR.exists():
                model_dir = DEFAULT_MODEL_DIR
            else:
                model_dir = FALLBACK_MODEL_DIR

        self.model_dir = Path(model_dir)
        self.threshold = threshold
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.eval()

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

    def predict_risk(self, steps: list) -> dict:
        """
        Given trajectory steps so far, returns risk score (probability of failure)
        and whether mid-course self-correction should be triggered.
        """
        text = self.serialize_steps(steps)
        if not text.strip():
            return {
                "risk_score": 0.0,
                "should_trigger_correction": False,
                "confidence": "low",
                "text_snippet": ""
            }

        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]
            failure_prob = float(probs[1].item())

        should_trigger = failure_prob >= self.threshold
        
        if failure_prob > 0.60:
            confidence = "high"
        elif failure_prob > 0.35:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "risk_score": round(failure_prob, 4),
            "should_trigger_correction": should_trigger,
            "confidence": confidence,
            "text_snippet": text[-200:]
        }

if __name__ == "__main__":
    predictor = FailurePredictor(threshold=0.35)
    sample_steps = [
        {"type": "human", "content": "Hi, my order 8891 was late and I want a refund now."},
        {"type": "tool_call", "tool": "issue_refund", "args": {"order_id": "8891", "amount": 49.99}},
        {"type": "tool_result", "content": "Refund issued."}
    ]
    res = predictor.predict_risk(sample_steps)
    print("Inference test result:", res)
