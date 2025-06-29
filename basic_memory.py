import math
from typing import Optional
from langchain.agents import initialize_agent,Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()
# 1. Initialize DeepSeek LLM
#    You may need to pass your DeepSeek API key or client config here.
llm = ChatDeepSeek(api_key=os.getenv("DEEPSEEK_API_KEY"), model="deepseek-chat")



def get_weather(location: str) -> str:
    # stub implementation—replace with real weather API call
    return f"Weather forecast for {location}: Sunny with a high of 25°C and low of 15°C. 10% chance of precipitation."



def calculator(operation_input: str) -> str:
    """Perform basic arithmetic operations from a text description."""
    try:
        # Parse the operation input
        parts = operation_input.split()
        
        # Try to identify the operation and numbers
        if "add" in operation_input.lower() or "+" in operation_input:
            # Look for two numbers in the string
            numbers = [float(s) for s in parts if s.replace('.', '', 1).isdigit()]
            if len(numbers) >= 2:
                return f"{numbers[0]} + {numbers[1]} = {numbers[0] + numbers[1]}"
            else:
                return "Error: Could not identify two numbers for addition"
            
        elif "subtract" in operation_input.lower() or "-" in operation_input:
            numbers = [float(s) for s in parts if s.replace('.', '', 1).isdigit()]
            if len(numbers) >= 2:
                return f"{numbers[0]} - {numbers[1]} = {numbers[0] - numbers[1]}"
            else:
                return "Error: Could not identify two numbers for subtraction"
            
        elif "multiply" in operation_input.lower() or "*" in operation_input:
            numbers = [float(s) for s in parts if s.replace('.', '', 1).isdigit()]
            if len(numbers) >= 2:
                return f"{numbers[0]} * {numbers[1]} = {numbers[0] * numbers[1]}"
            else:
                return "Error: Could not identify two numbers for multiplication"
            
        elif "divide" in operation_input.lower() or "/" in operation_input:
            numbers = [float(s) for s in parts if s.replace('.', '', 1).isdigit()]
            if len(numbers) >= 2:
                if numbers[1] == 0:
                    return "Error: Division by zero"
                return f"{numbers[0]} / {numbers[1]} = {numbers[0] / numbers[1]}"
            else:
                return "Error: Could not identify two numbers for division"
            
        elif "power" in operation_input.lower() or "^" in operation_input:
            numbers = [float(s) for s in parts if s.replace('.', '', 1).isdigit()]
            if len(numbers) >= 2:
                return f"{numbers[0]}^{numbers[1]} = {numbers[0] ** numbers[1]}"
            else:
                return "Error: Could not identify two numbers for power operation"
            
        elif "sqrt" in operation_input.lower():
            numbers = [float(s) for s in parts if s.replace('.', '', 1).isdigit()]
            if numbers:
                if numbers[0] < 0:
                    return "Error: Cannot take square root of negative number"
                return f"sqrt({numbers[0]}) = {math.sqrt(numbers[0])}"
            else:
                return "Error: Could not identify a number for square root"
        else:
            return f"Unknown operation. Please specify one of: add, subtract, multiply, divide, power, sqrt"
    except Exception as e:
        return f"Error performing calculation: {str(e)}"
    
    
tools = [
    Tool(
        name="Weather",
        func=get_weather,
        description="Get the current weather for a specified location."
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Perform mathematical calculations. Available operations: add, subtract, multiply, divide, power, sqrt. For example: operation='add', a=2, b=3 will return '2 + 3 = 5'."
    ),
]

# agent_executor = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
#     verbose=True,
# )

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
)

if __name__ == "__main__":
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        result = agent_executor.invoke(query)
        print(f"Result: {result['output']}")
