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
from langchain_core.callbacks import CallbackManager
from langfuse import Langfuse
from langfuse.model import CreateTrace
import subprocess
import json
from typing import Literal
from datetime import datetime

# Initialize Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "your_public_key"),  # Replace with your public key
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "your_secret_key"),  # Replace with your secret key
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# Define our state type
class AgentState(TypedDict):
    messages: List[BaseMessage]
    trace_id: str

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
    
    # Create a span for agent execution
    span = langfuse.span(
        name="agent_execution",
        trace_id=state["trace_id"]
    )
    span.start()
    
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
    
    # Log the response
    span.log(
        name="agent_response",
        level="INFO",
        metadata={
            "response_type": response.type,
            "has_function_call": bool(response.additional_kwargs.get("function_call")),
            "content": response.content
        }
    )
    
    span.end()
    return state

def should_continue(state: AgentState) -> Literal["continue", "final_answer"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # Create a span for decision point
    span = langfuse.span(
        name="decision_point",
        trace_id=state["trace_id"]
    )
    span.start()
    
    # If the last message contained a function call, continue
    has_function_call = isinstance(last_message, BaseMessage) and last_message.additional_kwargs.get("function_call")
    decision = "continue" if has_function_call else "final_answer"
    
    span.log(
        name="flow_decision",
        level="INFO",
        metadata={
            "decision": decision,
            "has_function_call": has_function_call
        }
    )
    
    span.end()
    return decision

def final_answer(state: AgentState) -> AgentState:
    span = langfuse.span(
        name="final_answer",
        trace_id=state["trace_id"]
    )
    span.start()
    
    span.log(
        name="final_response",
        level="INFO",
        metadata={
            "final_message": state["messages"][-1].content
        }
    )
    
    span.end()
    return state

class TrackedToolNode(ToolNode):
    def __call__(self, state):
        span = langfuse.span(
            name="tool_execution",
            trace_id=state["trace_id"]
        )
        span.start()
        
        # Execute the tool
        result = super().__call__(state)
        
        # Log the tool execution
        last_message = state["messages"][-1]
        if isinstance(last_message, BaseMessage) and last_message.additional_kwargs.get("function_call"):
            function_call = last_message.additional_kwargs["function_call"]
            span.log(
                name="tool_call",
                level="INFO",
                metadata={
                    "tool_name": function_call["name"],
                    "arguments": function_call["arguments"],
                    "result": str(result["messages"][-1].content)
                }
            )
        
        span.end()
        return result

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", run_agent)
workflow.add_node("tools", TrackedToolNode(tools))
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
    # Generate a unique trace ID for this conversation
    trace_id = f"conversation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Create a new trace
    trace = langfuse.trace(
        name="conversation",
        metadata={
            "initial_message": message,
            "timestamp": str(datetime.now())
        }
    )
    
    # Initialize the state with trace ID
    state = {
        "messages": [HumanMessage(content=message)],
        "trace_id": trace.id
    }
    
    # Run the graph
    result = app.invoke(state)
    
    # Log final state
    trace.log(
        name="conversation_complete",
        level="INFO",
        metadata={
            "message_count": len(result["messages"]),
            "final_message": result["messages"][-1].content
        }
    )
    
    trace.end()
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
            print(f"{message}:=========")
    
    # Close the Langfuse client
    langfuse.flush()
