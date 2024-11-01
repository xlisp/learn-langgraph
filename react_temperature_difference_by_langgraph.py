# Use LangGraph to replace the original AgentExecutor: react_temperature_difference_by_langchain.py

import operator
from typing import Annotated, TypedDict, Union
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import StateGraph, END

# Define global state variables for the state graph
class AgentState(TypedDict):
    # Accept user input
    input: str
    # Result of each Agent run, can be action, finish, or empty (initially)
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # Intermediate steps of Agent work, a sequence of actions and corresponding results
    # Declare the update of this state using append mode (instead of default overwrite) to retain intermediate steps
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

# Construct Agent node
def agent_node(state):
    outcome = agent.invoke(state)
    # Output needs to correspond to the keys in the state variables
    return {"agent_outcome": outcome}

# Construct tools node
def tools_node(state):
    # Identify the action from the Agent result
    agent_action = state["agent_outcome"]
    # Extract the corresponding tool from the action
    tool_to_use = {t.name: t for t in tools}[agent_action.tool]
    # Invoke the tool and get the result
    observation = tool_to_use.invoke(agent_action.tool_input)
    # Update the tool execution and result into the global state variable, as we declared its update mode, it will automatically append to the existing list
    return {"intermediate_steps": [(agent_action, observation)]}

# Initialize the state graph with global state variables
graph = StateGraph(AgentState)

# Add Agent and tools nodes respectively
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)

# Set the graph entry point
graph.set_entry_point("agent")

pdb.set_trace()

# Add conditional edges
graph.add_conditional_edges(
    # Starting point of conditional edge
    start_key="agent",
    # Condition judgment, we return different strings based on whether the Agent result is an action or finish
    condition=(
        lambda state: "exit"
        if isinstance(state["agent_outcome"], AgentFinish)
        else "continue"
    ),
    # Map the condition judgment result to the corresponding node
    conditional_edge_mapping={
        "continue": "tools",
        "exit": END,  # END is a special node indicating the graph's exit, terminating the run once reached
    },
)

# Don't forget to connect tools and Agent to ensure tool output returns to Agent for continued operation
graph.add_edge("tools", "agent")

# Generate a Runnable object for the graph
agent_graph = graph.compile()

# Invoke using the same interface as LCEL
agent_graph.invoke({"input": "What is the temperature difference between Shanghai and Beijing today?"})

