import os
import getpass

# Step 1: Set up API keys
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")

if "TAVILY_API_KEY" not in os.environ:
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter your Tavily API Key: ")

# Step 2: Import necessary modules
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Step 3: Initialize the Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Step 4: Initialize the Tavily Search tool
search_tool = TavilySearchResults(max_results=3)

# Step 5: Bind the tool to the model
model_with_tools = model.bind_tools([search_tool])

# Step 6: Test a simple prompt without requiring a tool
response = model_with_tools.invoke([HumanMessage(content="Hi!")])
print("== Non-tool response ==")
print(response.content)
print("Tool Calls:", response.tool_calls)

# Step 7: Test a prompt that requires the search tool
response = model_with_tools.invoke([HumanMessage(content="Latest Verizon news?")])
print("\n== Tool-assisted response ==")
print(response.content)
print("Tool Calls:", response.tool_calls)

# Step 8: Create a ReAct agent with the model and tool
agent_executor = create_react_agent(model, [search_tool])

# Step 9: Invoke the agent with a user message
response = agent_executor.invoke({"messages": [HumanMessage(content="What's the latest Verizon news?")]})
print("\n== Agent response ==")
for msg in response["messages"]:
    print(msg)