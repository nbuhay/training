"""
LangChain + Gemini + Tool Calling Demo
--------------------------------------

This script demonstrates how to integrate Google's Gemini model with LangChain,
define custom tools, and showcase how the model decides to invoke these tools
based on user prompts.

Prerequisites:
- Install required packages:
    pip install langchain langchain-google-genai

- Set your Google API key as an environment variable or input when prompted.
"""

import os
import getpass

# Step 1: Set up the Google API Key
# ---------------------------------
# Ensure the API key is set in the environment; if not, prompt the user.
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")

# Step 2: Import necessary modules from LangChain
# -----------------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# Step 3: Define a simple tool using the @tool decorator
# ------------------------------------------------------
# This tool multiplies two integers and returns the result.
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

# Step 4: Initialize the Gemini model
# -----------------------------------
# Create an instance of the ChatGoogleGenerativeAI with desired parameters.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Use "gemini-pro" for more advanced capabilities
    temperature=0.7,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Step 5: Bind the tool to the model
# ----------------------------------
# This allows the model to be aware of the tool and decide when to use it.
llm_with_tools = llm.bind_tools([multiply])

# Step 6: Demonstrate a prompt that doesn't require a tool
# --------------------------------------------------------
# The model should respond naturally without invoking any tools.
print("\n=== Example 1: Simple Greeting ===")
response = llm_with_tools.invoke([HumanMessage(content="Hello!")])
print(f"Model Response: {response.content}")
print(f"Tool Calls: {response.tool_calls}")  # Expecting: None or empty

# Step 7: Demonstrate a prompt that requires the tool
# ---------------------------------------------------
# The model should decide to invoke the 'multiply' tool.
print("\n=== Example 2: Multiplication Request ===")
response = llm_with_tools.invoke([HumanMessage(content="What is 6 times 7?")], tool_choice="multiply")
print(f"Model Response: {response.content}")
print(f"Tool Calls: {response.tool_calls}")  # Expecting: Details of the tool call

# Step 8: Execute the tool call manually and provide the result back to the model
# -------------------------------------------------------------------------------
# This simulates how an external system would handle the tool call.
if response.tool_calls:
    # Extract tool call details
    tool_call = response.tool_calls[0]
    print(tool_call)
    tool_name = tool_call['name']
    tool_args = tool_call['args']
    tool_call_id = tool_call['id']

    # Execute the tool based on the extracted arguments
    result = multiply.invoke(tool_args)
    print(result)

    # Create a ToolMessage with the result to send back to the model
    tool_message = ToolMessage(content=str(result), tool_call_id=tool_call_id)

    # Continue the conversation by providing the tool result
    follow_up = llm_with_tools.invoke([HumanMessage(content="What is 6 times 7?"), tool_message])
    print("\n=== Final Response After Tool Execution ===")
    print(follow_up)