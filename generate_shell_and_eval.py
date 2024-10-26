import os
from typing import Annotated, Sequence, TypedDict, Union, List
from langgraph.graph import Graph, MessageGraph
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.tools.base import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms.ollama import Ollama
import operator
import subprocess
from typing import List, Tuple
from pydantic import BaseModel, Field
import json

# Define our state type
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str

# Tool definitions
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

def run_shell_command(command: str) -> str:
    """Runs a shell command and returns the output or error"""
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        return e.stderr.decode('utf-8')

# Create structured tools
tools = [
    StructuredTool.from_function(multiply),
    StructuredTool.from_function(add),
    StructuredTool.from_function(run_shell_command)
]

# Initialize the LLM
llm = Ollama(model="llama3.1:latest") #, request_timeout=120.0)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant that can use tools to accomplish tasks.
    Your available tools are:
    {tool_descriptions}
    
    Use tools by specifying a json blob with "action" and "action_input".
    Think carefully about which tool to use and what input to provide.
    After using a tool, you'll get its output and can use that to plan next steps.
    
    Respond directly if no tools are needed."""),
    MessagesPlaceholder(variable_name="messages"),
])

# Create the tool executor
tool_executor = ToolExecutor(tools)

# Function to determine the next step
def should_continue(state: AgentState) -> Union[Tuple[str, AgentState], str]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message contained a function call, return "agent"
    if isinstance(last_message, FunctionMessage):
        return "agent"
    # If the last message was from the agent but didn't contain a function call,
    # then we're done
    return "end"

# Function to run the agent
def run_agent(state: AgentState) -> AgentState:
    messages = state["messages"]
    
    # Bind tools to the prompt
    function_descriptions = [convert_to_openai_function(t) for t in tools]
    prompt_with_tools = prompt.partial(
        tool_descriptions="\n".join([f"{t.name}: {t.description}" for t in tools])
    )
    
    # Generate agent's response
    response = llm.invoke(
        prompt_with_tools.format_messages(messages=messages),
        functions=function_descriptions
    )
    
    # Add agent's response to messages
    messages.append(response)
    return {"messages": messages, "next": "tool" if response.additional_kwargs.get("function_call") else "end"}

# Function to run tools
def run_tool(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    
    # Extract function call
    function_call = last_message.additional_kwargs["function_call"]
    action = function_call["name"]
    action_input = json.loads(function_call["arguments"])
    
    # Execute tool
    tool_result = tool_executor.invoke(action, action_input)
    
    # Add tool result to messages
    messages.append(
        FunctionMessage(content=str(tool_result), name=action)
    )
    return {"messages": messages, "next": "controller"}

# Build the graph
workflow = Graph()

# Add nodes
workflow.add_node("agent", run_agent)
workflow.add_node("tool", run_tool)
workflow.add_node("controller", should_continue)

# Add edges
workflow.add_edge("agent", "tool")
workflow.add_edge("tool", "controller")
workflow.add_edge("controller", "agent")
workflow.add_edge("controller", "end")

# Compile the graph
app = workflow.compile()

def chat(message: str) -> List[BaseMessage]:
    # Initialize the state
    state = {
        "messages": [HumanMessage(content=message)],
        "next": "agent"
    }
    
    # Run the graph
    result = app.invoke(state)
    return result["messages"]

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Please provide a question as an argument")
        sys.exit(1)
        
    question = sys.argv[1]
    messages = chat(question)
    
    # Print the final response
    for message in messages:
        if not isinstance(message, FunctionMessage):
            print(f"{message.type}: {message.content}")

