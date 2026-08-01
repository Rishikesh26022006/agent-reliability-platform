from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from tools import lookup_order, check_policy, issue_refund, escalate_to_human
from logger import log_trajectory

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

SYSTEM_PROMPT = """You are a customer support agent for an online store.
You have tools to look up orders, check refund policies, issue refunds, and escalate to a human.

Rules:
- Always look up the order first before doing anything else.
- Always cross-check any claim the customer makes (dates, delays, item condition, amounts) against
  the actual data returned by lookup_order. If the customer's claim contradicts the tool data
  (e.g. they say an order is late, but the delivery date hasn't passed yet), point this out
  and do not treat the claim as true.
- Always check the relevant policy before issuing a refund.
- Only issue a refund if the policy allows it.
- If you are unsure, or the situation seems like fraud, or the customer is upset about something
  outside these tools' scope, escalate to a human instead of guessing.
- Keep your final reply to the customer short, clear, and friendly.
"""

agent = create_react_agent(
    model=model,
    tools=[lookup_order, check_policy, issue_refund, escalate_to_human],
    prompt=SYSTEM_PROMPT,
)

def run_agent(user_query: str):
    result = agent.invoke({"messages": [HumanMessage(user_query)]})
    return result["messages"]


if __name__ == "__main__":
    query = "Hi, my order 8891 arrived broken. I'd like a refund please."
    messages = run_agent(query)

    traj_id = log_trajectory(query, messages, model_name="gemini-3.5-flash-lite")
    print(f"Logged trajectory: {traj_id}")

    for m in messages:
        print(f"--- {m.type} ---")
        print(m.content if m.content else m.additional_kwargs.get("function_call"))
        print()