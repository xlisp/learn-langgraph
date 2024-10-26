# Learn Langgraph

## Init

* Setup python env
```sh
conda create -n learn-langgraph python=3.11
conda activate learn-langgraph
poetry install
```
* [Ollama](https://ollama.com/) run llama3.1
```sh
ollama run llama3.1
```

## Basic ReActAgent Definition
```python
import os
from typing import Annotated, TypedDict, Union, List
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.tools.base import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms.ollama import Ollama
import subprocess
import json
from typing import Literal

# Define our state type
class AgentState(TypedDict):
    messages: List[BaseMessage]

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
llm = Ollama(model="llama3.1:latest")

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
    state["messages"].append(response)
    return state

def should_continue(state: AgentState) -> Literal["continue", "final_answer"]:
    messages = state["messages"]
    last_message = messages[-1]

    # If the last message contained a function call, continue
    if isinstance(last_message, BaseMessage) and last_message.additional_kwargs.get("function_call"):
        return "continue"
    # Otherwise, we're done
    return "final_answer"

def final_answer(state: AgentState) -> AgentState:
    return state

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", run_agent)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("final_answer", final_answer)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "final_answer": "final_answer"
    }
)

# Add edge from tools back to agent
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

def chat(message: str) -> List[BaseMessage]:
    # Initialize the state
    state = {"messages": [HumanMessage(content=message)]}

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
            print(f"{message} ==========")

# run success:
## (learn-langgraph) ➜  learn-langgraph git:(main) ✗ prunp generate_shell_and_eval.py "How many python files are there in the current directory?"
## /home/xlisp/EmacsPyPro/learn-langgraph/generate_shell_and_eval.py:44: LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaLLM``.
##   llm = Ollama(model="llama3.1:latest")
## content='How many python files are there in the current directory?' additional_kwargs={} response_metadata={} ==========
## How many python files are there in the current directory?
## I don't need to use any tools for this one. I can simply count the number of Python files in the current directory.
##
## There is 1 Python file in the current directory. ==========
##
```
