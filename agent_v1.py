from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
from tools import lookup_order

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
model_with_tools = model.bind_tools([lookup_order])

# Step 1: start the conversation
messages = [HumanMessage("Hi, can you check the status of my order? The order ID is 8891.")]

# Step 2: model decides to call a tool
ai_response = model_with_tools.invoke(messages)
messages.append(ai_response)

print("Step A - model requested:", ai_response.tool_calls)

# Step 3: WE actually run the tool (the model can't run code itself)
for tool_call in ai_response.tool_calls:
    if tool_call["name"] == "lookup_order":
        result = lookup_order(**tool_call["args"])
        print("Step B - tool ran, result:", result)

        # Step 4: feed the tool's result back to the model
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

# Step 5: model reads the tool result and writes the final customer-facing reply
final_response = model_with_tools.invoke(messages)
print("\nStep C - final reply to customer:")
print(final_response.content)