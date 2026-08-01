"""
Fallback policies and self-correction handlers when mid-trajectory failure predictor flags a high risk score.
"""

def get_correction_strategy(risk_score: float, category: str = "general") -> dict:
    """
    Determines the self-correction action to take based on the predicted risk score and failure category.
    """
    if risk_score >= 0.65:
        return {
            "action": "escalate",
            "message": "High failure risk detected (score >= 0.65). Escalating ticket to human support.",
            "prompt_injection": None
        }
    elif risk_score >= 0.35:
        return {
            "action": "reflect_and_verify",
            "message": "Elevated failure risk detected. Injecting reflection directive to verify claims against tool data.",
            "prompt_injection": (
                "SYSTEM AUDIT NOTICE: High risk of policy violation or unverified claim detected! "
                "You MUST call `lookup_order` to verify the actual order status, and call `check_policy` "
                "before taking any refund action. Do not accept customer claims without tool verification."
            )
        }
    else:
        return {
            "action": "proceed",
            "message": "Low risk.",
            "prompt_injection": None
        }
