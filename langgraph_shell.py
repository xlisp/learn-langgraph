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

# 配置日志

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(**name**)

class ActionType(Enum):
THINK = “think”
SHELL = “shell”
API_CALL = “api_call”
FINAL_ANSWER = “final_answer”

@dataclass
class ReActState:
“”“ReAct状态管理”””
messages: List[Dict[str, Any]] = field(default_factory=list)
current_step: int = 0
max_steps: int = 10
task: str = “”
thought: str = “”
action: str = “”
action_input: str = “”
observation: str = “”
final_answer: str = “”
shell_history: List[Dict[str, str]] = field(default_factory=list)
api_responses: List[Dict[str, Any]] = field(default_factory=list)

class ShellExecutor:
“”“安全的Shell执行器”””

```
def __init__(self, allowed_commands: List[str] = None):
    self.allowed_commands = allowed_commands or [
        'ls', 'pwd', 'cat', 'echo', 'grep', 'find', 'wc', 'head', 'tail',
        'curl', 'wget', 'ping', 'date', 'whoami', 'id', 'uname'
    ]

def execute(self, command: str, timeout: int = 30) -> Dict[str, Any]:
    """执行shell命令"""
    try:
        # 安全检查
        if not self._is_safe_command(command):
            return {
                "success": False,
                "output": "",
                "error": f"Command not allowed: {command.split()[0]}",
                "command": command
            }
        
        # 执行命令
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
    """检查命令是否安全"""
    cmd_parts = command.strip().split()
    if not cmd_parts:
        return False
    
    base_command = cmd_parts[0]
    
    # 检查是否在允许列表中
    if base_command not in self.allowed_commands:
        return False
    
    # 检查危险模式
    dangerous_patterns = [
        'rm -rf', 'sudo', 'su', 'chmod +x', 'wget', 'curl.*-o',
        '>', '>>', 'mkfifo', 'nc ', 'netcat'
    ]
    
    for pattern in dangerous_patterns:
        if pattern in command:
            return False
    
    return True
```

class APIClient:
“”“第三方API客户端”””

```
def __init__(self):
    self.base_headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'ReActAgent/1.0'
    }

def call_api(self, url: str, method: str = 'GET', 
             headers: Dict[str, str] = None, 
             data: Dict[str, Any] = None,
             params: Dict[str, Any] = None) -> Dict[str, Any]:
    """调用第三方API"""
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
```

class ReActAgent:
“”“ReAct智能体”””

```
def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
    self.llm = ChatOpenAI(api_key=api_key, model=model, temperature=0)
    self.shell_executor = ShellExecutor()
    self.api_client = APIClient()
    
    # 创建工具
    self.tools = [
        self._create_shell_tool(),
        self._create_api_tool(),
        self._create_think_tool()
    ]
    
    # 创建图
    self.graph = self._create_graph()

def _create_shell_tool(self):
    """创建shell执行工具"""
    @tool
    def execute_shell(command: str) -> str:
        """Execute a shell command safely"""
        result = self.shell_executor.execute(command)
        return json.dumps(result, indent=2)
    
    return execute_shell

def _create_api_tool(self):
    """创建API调用工具"""
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
    """创建思考工具"""
    @tool
    def think(thought: str) -> str:
        """Record a thought or reasoning step"""
        return f"Thought recorded: {thought}"
    
    return think

def _create_graph(self) -> StateGraph:
    """创建LangGraph工作流"""
    workflow = StateGraph(ReActState)
    
    # 添加节点
    workflow.add_node("reasoning", self._reasoning_node)
    workflow.add_node("action", self._action_node)
    workflow.add_node("observation", self._observation_node)
    workflow.add_node("final_answer", self._final_answer_node)
    
    # 添加边
    workflow.add_edge("reasoning", "action")
    workflow.add_edge("action", "observation")
    workflow.add_conditional_edges(
        "observation",
        self._should_continue,
        {
            "continue": "reasoning",
            "final_answer": "final_answer",
            "end": END
        }
    )
    workflow.add_edge("final_answer", END)
    
    # 设置入口点
    workflow.set_entry_point("reasoning")
    
    return workflow.compile()

def _reasoning_node(self, state: ReActState) -> ReActState:
    """推理节点"""
    logger.info(f"Step {state.current_step}: Reasoning")
    
    # 构建推理提示
    prompt = self._build_reasoning_prompt(state)
    
    # 调用LLM进行推理
    response = self.llm.invoke([SystemMessage(content=prompt)])
    
    # 解析响应
    thought, action, action_input = self._parse_reasoning_response(response.content)
    
    state.thought = thought
    state.action = action
    state.action_input = action_input
    
    logger.info(f"Thought: {thought}")
    logger.info(f"Action: {action}")
    logger.info(f"Action Input: {action_input}")
    
    return state

def _action_node(self, state: ReActState) -> ReActState:
    """行动节点"""
    logger.info(f"Step {state.current_step}: Action - {state.action}")
    
    if state.action == "shell":
        result = self.shell_executor.execute(state.action_input)
        state.shell_history.append({
            "command": state.action_input,
            "result": result
        })
        state.observation = json.dumps(result, indent=2)
        
    elif state.action == "api_call":
        try:
            api_params = json.loads(state.action_input)
            result = self.api_client.call_api(**api_params)
            state.api_responses.append(result)
            state.observation = json.dumps(result, indent=2)
        except Exception as e:
            state.observation = f"API call failed: {str(e)}"
    
    elif state.action == "think":
        state.observation = f"Thought: {state.action_input}"
    
    elif state.action == "final_answer":
        state.final_answer = state.action_input
        state.observation = "Final answer provided"
    
    else:
        state.observation = f"Unknown action: {state.action}"
    
    return state

def _observation_node(self, state: ReActState) -> ReActState:
    """观察节点"""
    logger.info(f"Step {state.current_step}: Observation")
    logger.info(f"Observation: {state.observation}")
    
    # 记录消息
    state.messages.append({
        "step": state.current_step,
        "thought": state.thought,
        "action": state.action,
        "action_input": state.action_input,
        "observation": state.observation
    })
    
    state.current_step += 1
    
    return state

def _final_answer_node(self, state: ReActState) -> ReActState:
    """最终答案节点"""
    logger.info(f"Final Answer: {state.final_answer}")
    return state

def _should_continue(self, state: ReActState) -> str:
    """决定是否继续"""
    if state.action == "final_answer":
        return "final_answer"
    elif state.current_step >= state.max_steps:
        return "end"
    else:
        return "continue"

def _build_reasoning_prompt(self, state: ReActState) -> str:
    """构建推理提示"""
    prompt = f"""
```

You are a ReAct agent that can think, execute shell commands, and call APIs.

Task: {state.task}

You have access to the following actions:

1. think: Record your thoughts and reasoning
1. shell: Execute shell commands (safe commands only)
1. api_call: Call third-party APIs
1. final_answer: Provide the final answer

Current step: {state.current_step}/{state.max_steps}

Previous steps:
“””

```
    for msg in state.messages[-3:]:  # Show last 3 steps
        prompt += f"""
```

Step {msg[‘step’]}:
Thought: {msg[‘thought’]}
Action: {msg[‘action’]}
Action Input: {msg[‘action_input’]}
Observation: {msg[‘observation’]}
“””

```
    prompt += """
```

Please provide your response in the following format:
Thought: [your reasoning about what to do next]
Action: [one of: think, shell, api_call, final_answer]
Action Input: [input for the action]

If using shell action, provide the command to execute.
If using api_call action, provide JSON with parameters: {“url”: “…”, “method”: “GET”, “headers”: {…}, “data”: {…}}
If using final_answer action, provide your final response to the task.
“””

```
    return prompt

def _parse_reasoning_response(self, response: str) -> tuple:
    """解析推理响应"""
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
    """运行ReAct智能体"""
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
```

# 使用示例

async def main():
# 配置API密钥
OPENAI_API_KEY = os.getenv(“OPENAI_API_KEY”, “your-api-key-here”)

```
# 创建ReAct智能体
agent = ReActAgent(api_key=OPENAI_API_KEY)

# 示例任务
tasks = [
    "检查当前目录下的文件，然后调用JSONPlaceholder API获取用户信息",
    "获取系统信息并调用天气API查询当前天气",
    "查找包含'python'的文件，然后调用GitHub API搜索相关仓库"
]

for task in tasks:
    print(f"\n{'='*50}")
    print(f"Task: {task}")
    print('='*50)
    
    result = agent.run(task, max_steps=8)
    
    if result["success"]:
        print(f"Final Answer: {result['final_answer']}")
        print(f"Steps taken: {result['steps']}")
        
        # 显示执行历史
        print("\nExecution History:")
        for i, msg in enumerate(result["messages"], 1):
            print(f"\nStep {i}:")
            print(f"  Thought: {msg['thought']}")
            print(f"  Action: {msg['action']}")
            print(f"  Input: {msg['action_input']}")
            print(f"  Observation: {msg['observation'][:200]}...")
    else:
        print(f"Error: {result['error']}")
```

if **name** == “**main**”:
asyncio.run(main())