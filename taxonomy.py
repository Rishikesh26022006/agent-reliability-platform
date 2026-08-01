FAILURE_CATEGORIES = {
    "wrong_tool_selection": "Called the wrong tool for the situation (e.g. escalated when a direct action was correct, or vice versa).",
    "hallucinated_policy": "Stated a policy, explanation, or justification not supported by any tool output — including inventing plausible-sounding excuses to explain away contradicting data.",
    "incomplete_resolution": "Stopped before finishing the task, left the customer's request partially unaddressed.",
    "premature_escalation": "Escalated a simple, clearly resolvable case unnecessarily.",
    "missed_escalation": "Failed to escalate a case that clearly warranted human review (fraud, legal threat, safety issue).",
    "tool_argument_error": "Called the right tool but with incorrect parameters (wrong order ID, wrong amount, etc).",
    "unverified_claim": "Accepted a customer's factual claim (dates, amounts, condition) without cross-checking it against available tool data, and acted on the unverified claim.",
    "inconsistent_policy_execution": "Handled two instances of the same policy category (e.g. change_of_mind, accidental damage) with different outcomes, with no stated justification for the difference.",
    "premature_resolution_without_verification": "Granted a refund or resolution based on vague/emotional pressure rather than a specific, verifiable policy trigger.",
    "response_data_mismatch": "The tool call itself was correct, but the natural-language reply to the customer contains information (order ID, amount, item name) that doesn't match the actual tool output.",
}

def format_taxonomy_for_prompt() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in FAILURE_CATEGORIES.items())