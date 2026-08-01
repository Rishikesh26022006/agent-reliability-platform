import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent_graph import agent
from failure_predictor.inference import FailurePredictor
from correction.fallback_policies import get_correction_strategy
from logger import log_trajectory

class ReliableAgentRunner:
    def __init__(self, risk_threshold: float = 0.35):
        self.predictor = FailurePredictor(threshold=risk_threshold)
        self.agent = agent

    def parse_messages_to_steps(self, messages: list) -> list:
        steps = []
        for m in messages:
            if m.type == "human":
                steps.append({"type": "human", "content": m.content})
            elif m.type == "ai":
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        steps.append({"type": "tool_call", "tool": tc["name"], "args": tc["args"]})
                if m.content:
                    steps.append({"type": "ai_response", "content": m.content})
            elif m.type == "tool":
                steps.append({"type": "tool_result", "content": m.content})
        return steps

    def run_with_guardrails(self, user_query: str) -> dict:
        messages = [HumanMessage(content=user_query)]
        corrections_triggered = []
        
        # Invoke initial agent step
        result = self.agent.invoke({"messages": messages})
        current_messages = result["messages"]

        steps = self.parse_messages_to_steps(current_messages)
        risk_info = self.predictor.predict_risk(steps)

        if risk_info["should_trigger_correction"]:
            strategy = get_correction_strategy(risk_info["risk_score"])
            corrections_triggered.append({
                "risk_score": risk_info["risk_score"],
                "strategy": strategy["action"],
                "message": strategy["message"]
            })

            if strategy["action"] == "reflect_and_verify" and strategy["prompt_injection"]:
                # Re-prompt agent with audit directive
                injection_msg = HumanMessage(content=strategy["prompt_injection"])
                audit_result = self.agent.invoke({"messages": current_messages + [injection_msg]})
                current_messages = audit_result["messages"]

        traj_id = log_trajectory(user_query, current_messages, model_name="gemini-3.5-flash-lite")

        return {
            "trajectory_id": traj_id,
            "query": user_query,
            "final_messages": current_messages,
            "final_response": current_messages[-1].content if current_messages else "",
            "risk_info": risk_info,
            "corrections_triggered": corrections_triggered
        }

if __name__ == "__main__":
    runner = ReliableAgentRunner(risk_threshold=0.35)
    res = runner.run_with_guardrails("Hi, my order 8891 arrived broken. I'd like a refund please.")
    print("Reliable Agent Run Output:")
    print("Trajectory ID:", res["trajectory_id"])
    print("Risk Info:", res["risk_info"])
    print("Corrections Triggered:", res["corrections_triggered"])
    print("Final Response:", res["final_response"])
