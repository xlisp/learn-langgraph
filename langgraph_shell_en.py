import asyncio
import subprocess
import json
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionType(Enum):
    THINK = "think"
    SHELL = "shell"
    API_CALL = "api_call"
    FINAL_ANSWER = "final_answer"

@dataclass
class ReActState:
    """ReAct state management"""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    max_steps: int = 10
    task: str = ""
    thought: str = ""
    current_action: str = ""
    action_input: str = ""
    observation: str = ""
    final_answer: str = ""
    shell_history: List[Dict[str, str]] = field(default_factory=list)
    api_responses: List[Dict[str, Any]] = field(default_factory=list)

class ShellExecutor:
    """Safe shell command executor"""
    
    def __init__(self, allowed_commands: List[str] = None):
        self.allowed_commands = allowed_commands or [
            'ls', 'pwd', 'cat', 'echo', 'grep', 'find', 'wc', 'head', 'tail',
            'curl', 'wget', 'ping', 'date', 'whoami', 'id', 'uname', 'ps',
            'top', 'df', 'du', 'free', 'uptime', 'which', 'whereis'
        ]
    
    def execute(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute shell command safely"""
        try:
            # Safety check
            if not self._is_safe_command(command):
                return {
                    "success": False,
                    "output": "",
                    "error": f"Command not allowed: {command.split()[0]}",
                    "command": command
                }
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "command": command,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Command timed out after {timeout} seconds",
                "command": command
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "command": command
            }
    
    def _is_safe_command(self, command: str) -> bool:
        """Check if command is safe to execute"""
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return False
        
        base_command = cmd_parts[0]
        
        # Check if command is in allowed list
        if base_command not in self.allowed_commands:
            return False
        
        # Check for dangerous patterns
        dangerous_patterns = [
            'rm -rf', 'sudo', 'su', 'chmod +x', 'wget.*-o', 'curl.*-o',
            '>', '>>', 'mkfifo', 'nc ', 'netcat', 'dd ', 'format',
            'fdisk', 'mount', 'umount', 'kill', 'killall'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in command:
                return False
        
        return True

class APIClient:
    """Third-party API client"""
    
    def __init__(self):
        self.base_headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'ReActAgent/1.0'
        }
    
    def call_api(self, url: str, method: str = 'GET', 
                 headers: Dict[str, str] = None, 
                 data: Dict[str, Any] = None,
                 params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call third-party API"""
        try:
            request_headers = {**self.base_headers}
            if headers:
                request_headers.update(headers)
            
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=request_headers,
                json=data,
                params=params,
                timeout=30
            )
            
            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "data": response.json() if response.content else None,
                "url": url,
                "method": method
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "method": method
            }
        except json.JSONDecodeError:
            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "data": response.text,
                "url": url,
                "method": method
            }

class ReActAgent:
    """ReAct intelligent agent"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(api_key=api_key, model=model, temperature=0)
        self.shell_executor = ShellExecutor()
        self.api_client = APIClient()
        
        # Create tools
        self.tools = [
            self._create_shell_tool(),
            self._create_api_tool(),
            self._create_think_tool()
        ]
        
        # Create graph
        self.graph = self._create_graph()
    
    def _create_shell_tool(self):
        """Create shell execution tool"""
        @tool
        def execute_shell(command: str) -> str:
            """Execute a shell command safely"""
            result = self.shell_executor.execute(command)
            return json.dumps(result, indent=2)
        
        return execute_shell
    
    def _create_api_tool(self):
        """Create API calling tool"""
        @tool
        def call_api(url: str, method: str = "GET", 
                    headers: str = None, data: str = None, 
                    params: str = None) -> str:
            """Call a third-party API"""
            try:
                headers_dict = json.loads(headers) if headers else None
                data_dict = json.loads(data) if data else None
                params_dict = json.loads(params) if params else None
                
                result = self.api_client.call_api(
                    url=url,
                    method=method,
                    headers=headers_dict,
                    data=data_dict,
                    params=params_dict
                )
                return json.dumps(result, indent=2)
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})
        
        return call_api
    
    def _create_think_tool(self):
        """Create thinking tool"""
        @tool
        def think(thought: str) -> str:
            """Record a thought or reasoning step"""
            return f"Thought recorded: {thought}"
        
        return think
    
    def _create_graph(self) -> StateGraph:
        """Create LangGraph workflow"""
        workflow = StateGraph(ReActState)
        
        # Add nodes - using different names to avoid conflicts with state keys
        workflow.add_node("reasoning", self._reasoning_node)
        workflow.add_node("execute_action", self._action_node)
        workflow.add_node("observe", self._observation_node)
        workflow.add_node("provide_answer", self._final_answer_node)
        
        # Add edges
        workflow.add_edge("reasoning", "execute_action")
        workflow.add_edge("execute_action", "observe")
        workflow.add_conditional_edges(
            "observe",
            self._should_continue,
            {
                "continue": "reasoning",
                "final_answer": "provide_answer",
                "end": END
            }
        )
        workflow.add_edge("provide_answer", END)
        
        # Set entry point
        workflow.set_entry_point("reasoning")
        
        return workflow.compile()
    
    def _reasoning_node(self, state: ReActState) -> ReActState:
        """Reasoning node"""
        logger.info(f"Step {state.current_step}: Reasoning")
        
        # Build reasoning prompt
        prompt = self._build_reasoning_prompt(state)
        
        # Call LLM for reasoning
        response = self.llm.invoke([SystemMessage(content=prompt)])
        
        # Parse response
        thought, action, action_input = self._parse_reasoning_response(response.content)
        
        state.thought = thought
        state.current_action = action
        state.action_input = action_input
        
        logger.info(f"Thought: {thought}")
        logger.info(f"Action: {action}")
        logger.info(f"Action Input: {action_input}")
        
        return state
    
    def _action_node(self, state: ReActState) -> ReActState:
        """Action node"""
        logger.info(f"Step {state.current_step}: Action - {state.current_action}")
        
        if state.current_action == "shell":
            result = self.shell_executor.execute(state.action_input)
            state.shell_history.append({
                "command": state.action_input,
                "result": result
            })
            state.observation = json.dumps(result, indent=2)
            
        elif state.current_action == "api_call":
            try:
                api_params = json.loads(state.action_input)
                result = self.api_client.call_api(**api_params)
                state.api_responses.append(result)
                state.observation = json.dumps(result, indent=2)
            except Exception as e:
                state.observation = f"API call failed: {str(e)}"
        
        elif state.current_action == "think":
            state.observation = f"Thought: {state.action_input}"
        
        elif state.current_action == "final_answer":
            state.final_answer = state.action_input
            state.observation = "Final answer provided"
        
        else:
            state.observation = f"Unknown action: {state.current_action}"
        
        return state
    
    def _observation_node(self, state: ReActState) -> ReActState:
        """Observation node"""
        logger.info(f"Step {state.current_step}: Observation")
        logger.info(f"Observation: {state.observation}")
        
        # Record message
        state.messages.append({
            "step": state.current_step,
            "thought": state.thought,
            "action": state.current_action,
            "action_input": state.action_input,
            "observation": state.observation
        })
        
        state.current_step += 1
        
        return state
    
    def _final_answer_node(self, state: ReActState) -> ReActState:
        """Final answer node"""
        logger.info(f"Final Answer: {state.final_answer}")
        return state
    
    def _should_continue(self, state: ReActState) -> str:
        """Decide whether to continue"""
        if state.current_action == "final_answer":
            return "final_answer"
        elif state.current_step >= state.max_steps:
            return "end"
        else:
            return "continue"
    
    def _build_reasoning_prompt(self, state: ReActState) -> str:
        """Build reasoning prompt"""
        prompt = f"""
You are a ReAct agent that can think, execute shell commands, and call APIs.

Task: {state.task}

You have access to the following actions:
1. think: Record your thoughts and reasoning
2. shell: Execute shell commands (safe commands only)
3. api_call: Call third-party APIs
4. final_answer: Provide the final answer

Available shell commands: ls, pwd, cat, echo, grep, find, wc, head, tail, curl, wget, ping, date, whoami, id, uname, ps, top, df, du, free, uptime, which, whereis

Current step: {state.current_step}/{state.max_steps}

Previous steps:
"""
        
        for msg in state.messages[-3:]:  # Show last 3 steps
            prompt += f"""
Step {msg['step']}:
Thought: {msg['thought']}
Action: {msg['action']}
Action Input: {msg['action_input']}
Observation: {msg['observation']}
"""
        
        prompt += """
Please provide your response in the following format:
Thought: [your reasoning about what to do next]
Action: [one of: think, shell, api_call, final_answer]
Action Input: [input for the action]

Guidelines:
- If using shell action, provide the command to execute
- If using api_call action, provide JSON with parameters: {"url": "...", "method": "GET", "headers": {...}, "data": {...}}
- If using final_answer action, provide your final response to the task
- Be systematic and break down complex tasks into smaller steps
- Always think before taking action
- Use shell commands to gather information about the system
- Use API calls to fetch external data
"""
        
        return prompt
    
    def _parse_reasoning_response(self, response: str) -> tuple:
        """Parse reasoning response"""
        lines = response.strip().split('\n')
        thought = ""
        action = ""
        action_input = ""
        
        for line in lines:
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                action_input = line.replace("Action Input:", "").strip()
        
        return thought, action, action_input
    
    def run(self, task: str, max_steps: int = 10) -> Dict[str, Any]:
        """Run ReAct agent"""
        logger.info(f"Starting ReAct agent with task: {task}")
        
        initial_state = ReActState(
            task=task,
            max_steps=max_steps
        )
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            return {
                "success": True,
                "task": task,
                "final_answer": final_state.final_answer,
                "steps": final_state.current_step,
                "messages": final_state.messages,
                "shell_history": final_state.shell_history,
                "api_responses": final_state.api_responses
            }
            
        except Exception as e:
            logger.error(f"Error running ReAct agent: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task": task
            }

# Usage examples
async def main():
    """Main function demonstrating ReAct agent usage"""
    # Configure API key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    
    # Create ReAct agent
    agent = ReActAgent(api_key=OPENAI_API_KEY)
    
    # Example tasks
    tasks = [
        "Check the current directory files and call JSONPlaceholder API to get user information",
        "Get system information and call a weather API to check current weather",
        "Find files containing 'python' keyword and call GitHub API to search related repositories",
        "Check disk usage and call a public API to get current time information",
        "List running processes and call a news API to get latest headlines"
    ]
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print('='*60)
        
        result = agent.run(task, max_steps=8)
        
        if result["success"]:
            print(f"\nFinal Answer: {result['final_answer']}")
            print(f"Steps taken: {result['steps']}")
            
            # Show execution history
            print("\nExecution History:")
            for i, msg in enumerate(result["messages"], 1):
                print(f"\nStep {i}:")
                print(f"  Thought: {msg['thought']}")
                print(f"  Action: {msg['action']}")
                print(f"  Input: {msg['action_input']}")
                print(f"  Observation: {msg['observation'][:200]}{'...' if len(msg['observation']) > 200 else ''}")
            
            # Show shell commands used
            if result["shell_history"]:
                print("\nShell Commands Executed:")
                for cmd in result["shell_history"]:
                    print(f"  Command: {cmd['command']}")
                    print(f"  Success: {cmd['result']['success']}")
                    if cmd['result']['output']:
                        print(f"  Output: {cmd['result']['output'][:100]}{'...' if len(cmd['result']['output']) > 100 else ''}")
            
            # Show API calls made
            if result["api_responses"]:
                print("\nAPI Calls Made:")
                for api_call in result["api_responses"]:
                    print(f"  URL: {api_call['url']}")
                    print(f"  Method: {api_call['method']}")
                    print(f"  Success: {api_call['success']}")
                    if api_call.get('status_code'):
                        print(f"  Status: {api_call['status_code']}")
        else:
            print(f"Error: {result['error']}")
        
        print("\n" + "="*60 + "\n")

# Additional utility functions
def create_custom_agent(api_key: str, allowed_commands: List[str] = None, 
                       custom_tools: List[Any] = None) -> ReActAgent:
    """Create a custom ReAct agent with specific configurations"""
    agent = ReActAgent(api_key=api_key)
    
    # Override allowed commands if provided
    if allowed_commands:
        agent.shell_executor.allowed_commands = allowed_commands
    
    # Add custom tools if provided
    if custom_tools:
        agent.tools.extend(custom_tools)
    
    return agent

def run_interactive_mode(agent: ReActAgent):
    """Run agent in interactive mode"""
    print("ReAct Agent Interactive Mode")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        task = input("\nEnter task: ").strip()
        
        if task.lower() in ['quit', 'exit']:
            break
        
        if not task:
            continue
        
        result = agent.run(task)
        
        if result["success"]:
            print(f"\nAnswer: {result['final_answer']}")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    # Run main demo
    asyncio.run(main())
    
    # Uncomment to run interactive mode
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    # agent = ReActAgent(api_key=OPENAI_API_KEY)
    # run_interactive_mode(agent)

# @ python langgraph_shell_en.py
# /opt/anaconda3/lib/python3.11/site-packages/langgraph/graph/graph.py:31: LangChainDeprecationWarning: As of langchain-core 0.3.0, LangChain uses pydantic v2 internally. The langchain_core.pydantic_v1 module was a compatibility shim for pydantic v1, and should no longer be used. Please update the code to import from Pydantic directly.
# 
# For example, replace imports like: `from langchain_core.pydantic_v1 import BaseModel`
# with: `from pydantic import BaseModel`
# or the v1 compatibility namespace if you are working in a code base that has not been fully upgraded to pydantic 2 yet. 	from pydantic.v1 import BaseModel
# 
#   from langgraph.pregel import Channel, Pregel
# 
# ============================================================
# Task: Check the current directory files and call JSONPlaceholder API to get user information
# ============================================================
# INFO:__main__:Starting ReAct agent with task: Check the current directory files and call JSONPlaceholder API to get user information
# INFO:__main__:Step 0: Reasoning
# INFO:httpx:HTTP Request: POST https://openrouter.ai/api/chat/completions "HTTP/1.1 200 OK"
# ERROR:__main__:Error running ReAct agent: 'str' object has no attribute 'model_dump'
# Error: 'str' object has no attribute 'model_dump'
# 
# ============================================================
# 
# 
# ============================================================
# Task: Get system information and call a weather API to check current weather
# ============================================================
# INFO:__main__:Starting ReAct agent with task: Get system information and call a weather API to check current weather
# INFO:__main__:Step 0: Reasoning
# INFO:httpx:HTTP Request: POST https://openrouter.ai/api/chat/completions "HTTP/1.1 200 OK"
# ERROR:__main__:Error running ReAct agent: 'str' object has no attribute 'model_dump'
# Error: 'str' object has no attribute 'model_dump'
# 
# ============================================================
# 
# 
# ============================================================
# Task: Find files containing 'python' keyword and call GitHub API to search related repositories
# ============================================================
# INFO:__main__:Starting ReAct agent with task: Find files containing 'python' keyword and call GitHub API to search related repositories
# INFO:__main__:Step 0: Reasoning
# INFO:httpx:HTTP Request: POST https://openrouter.ai/api/chat/completions "HTTP/1.1 200 OK"
# ERROR:__main__:Error running ReAct agent: 'str' object has no attribute 'model_dump'
# Error: 'str' object has no attribute 'model_dump'
# 
# ============================================================
# 
# 
# ============================================================
# Task: Check disk usage and call a public API to get current time information
# ============================================================
# INFO:__main__:Starting ReAct agent with task: Check disk usage and call a public API to get current time information
# INFO:__main__:Step 0: Reasoning
# INFO:httpx:HTTP Request: POST https://openrouter.ai/api/chat/completions "HTTP/1.1 200 OK"
# ERROR:__main__:Error running ReAct agent: 'str' object has no attribute 'model_dump'
# Error: 'str' object has no attribute 'model_dump'
# 
# ============================================================
# 
# 
# ============================================================
# Task: List running processes and call a news API to get latest headlines
# ============================================================
# INFO:__main__:Starting ReAct agent with task: List running processes and call a news API to get latest headlines
# INFO:__main__:Step 0: Reasoning
# INFO:httpx:HTTP Request: POST https://openrouter.ai/api/chat/completions "HTTP/1.1 200 OK"
# ERROR:__main__:Error running ReAct agent: 'str' object has no attribute 'model_dump'
# Error: 'str' object has no attribute 'model_dump'
# 
# ============================================================
# 
# 坚持去λ化(中-易) learn-langgraph  main @ gs
# 
