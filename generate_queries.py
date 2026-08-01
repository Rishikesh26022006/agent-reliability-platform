from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")  # stronger model for higher-quality generation

GENERATION_PROMPT = """Generate 40 diverse customer support queries for an e-commerce refund/order support agent.

Context - available fake orders in the system:
- Order 8891: Wireless Mouse, $24.99, delivered 2026-07-18
- Order 4432: Bluetooth Headphones, $79.99, in_transit, expected 2026-07-25
- Order 1029: Laptop Stand, $34.50, delivered 2026-07-15
(Today's date is 2026-07-23)

Mix in these types roughly evenly:
- Simple, clear-cut requests (damaged item, wrong item, status check)
- Ambiguous or multi-issue requests (customer mentions two problems at once)
- Edge cases (order ID that doesn't exist, e.g. order 5555 or 2201)
- Emotionally charged / angry customers
- Requests with claims that CONTRADICT the actual order data above (e.g. claiming a delivered
  item hasn't arrived, or claiming a not-yet-due order is late)
- Fraud/security concerns
- Vague or incomplete requests that don't give an order ID
- Requests that try to get a refund not actually justified by policy (testing if agent pushes back)

Return ONLY a JSON array of 40 strings, no other text, no markdown formatting, no code fences.
"""


def extract_text(response) -> str:
    """Handle both possible response formats: plain string vs list of content blocks."""
    if isinstance(response.content, str):
        return response.content.strip()
    return " ".join(
        block.get("text", "") for block in response.content if isinstance(block, dict)
    ).strip()


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences if the model added them despite instructions not to."""
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


if __name__ == "__main__":
    response = model.invoke(GENERATION_PROMPT)

    text = extract_text(response)
    text = strip_code_fences(text)

    try:
        queries = json.loads(text)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON. Raw model output was:")
        print(text)
        raise e

    print(f"Generated {len(queries)} queries")

    with open("data/synthetic_queries.json", "w") as f:
        json.dump(queries, f, indent=2)

    print("Saved to data/synthetic_queries.json")
    for q in queries[:5]:
        print("-", q)