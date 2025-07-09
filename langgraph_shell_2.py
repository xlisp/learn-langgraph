"""
LangGraph ReAct Shell执行器 - 结合第三方API (修复版)
支持推理、行动、观察循环，执行shell命令和API调用
"""

import json
import subprocess
import requests
from typing import Dict, Any, List, Optional, Literal, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
import os
from datetime import datetime

# 配置
OPENAI_API_KEY = "your-openai-api-key"  # 请替换为您的API密钥
WEATHER_API_KEY = "your-weather-api-key"  # 请替换为您的天气API密钥

# 定义状态类型
class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_step: int
    max_steps: int
    task_completed: bool
    shell_history: List[Dict[str, str]]
    api_calls: List[Dict[str, Any]]

# 工具定义
@tool
def execute_shell_command(command: str) -> str:
    """
    执行shell命令
    
    Args:
        command: 要执行的shell命令
    
    Returns:
        命令执行结果
    """
    try:
        # 安全检查 - 避免危险命令
        dangerous_commands = ['rm -rf', 'sudo rm', 'chmod +x', 'wget', 'curl -X DELETE', 'dd if=']
        if any(dangerous in command.lower() for dangerous in dangerous_commands):
            return f"错误: 命令 '{command}' 被安全策略阻止"
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = f"退出码: {result.returncode}\n"
        if result.stdout:
            output += f"标准输出:\n{result.stdout}\n"
        if result.stderr:
            output += f"标准错误:\n{result.stderr}\n"
        
        return output
    
    except subprocess.TimeoutExpired:
        return f"错误: 命令执行超时 (30秒)"
    except Exception as e:
        return f"错误: {str(e)}"

@tool
def get_weather_info(city: str) -> str:
    """
    获取天气信息
    
    Args:
        city: 城市名称
    
    Returns:
        天气信息JSON字符串
    """
    try:
        # 模拟天气API调用（因为需要真实API密钥）
        # 实际使用时请替换为真实的API调用
        weather_data = {
            'city': city,
            'temperature': 25.5,
            'description': '晴天',
            'humidity': 60,
            'pressure': 1013
        }
        return json.dumps(weather_data, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"错误: {str(e)}"

@tool
def search_github_repos(query: str) -> str:
    """
    搜索GitHub仓库
    
    Args:
        query: 搜索关键词
    
    Returns:
        搜索结果JSON字符串
    """
    try:
        url = "https://api.github.com/search/repositories"
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': 5
        }
        
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'LangGraph-ReAct-Agent'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            repos = []
            for item in data['items']:
                repos.append({
                    'name': item['name'],
                    'full_name': item['full_name'],
                    'description': item['description'],
                    'stars': item['stargazers_count'],
                    'language': item['language'],
                    'url': item['html_url']
                })
            return json.dumps(repos, ensure_ascii=False, indent=2)
        else:
            return f"错误: 搜索失败，状态码: {response.status_code}"
    
    except Exception as e:
        return f"错误: {str(e)}"

@tool
def get_system_info() -> str:
    """
    获取系统信息
    
    Returns:
        系统信息字符串
    """
    try:
        import platform
        
        info = {
            'system': platform.system(),
            'node': platform.node(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
        
        return json.dumps(info, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"错误: {str(e)}"

# 定义所有工具
tools = [execute_shell_command, get_weather_info, search_github_repos, get_system_info]

class ReActShellAgent:
    """ReAct Shell执行代理"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY
        )
        self.tools = tools
        self.tool_executor = ToolExecutor(tools)
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """创建LangGraph工作流"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("action", self._action_node)
        
        # 设置入口点
        workflow.set_entry_point("agent")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "action",
                "end": END
            }
        )
        
        # 从action回到agent
        workflow.add_edge("action", "agent")
        
        return workflow.compile()
    
    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Agent推理节点"""
        print(f"\n=== Agent推理 (步骤 {state['current_step']}) ===")
        
        # 构建系统消息
        system_msg = SystemMessage(content="""你是一个能够执行shell命令和调用API的智能助手。

可用工具:
- execute_shell_command: 执行shell命令
- get_weather_info: 获取天气信息  
- search_github_repos: 搜索GitHub仓库
- get_system_info: 获取系统信息

请按ReAct模式工作:
1. 思考当前需要做什么
2. 决定调用哪个工具
3. 如果任务完成，回复"FINISHED: [总结]"

请直接调用工具，不要用JSON格式回复。""")
        
        # 构建消息列表
        messages = [system_msg] + state["messages"]
        
        # 调用LLM
        response = self.llm.bind_tools(self.tools).invoke(messages)
        
        print(f"Agent回复: {response.content}")
        
        # 更新状态
        new_state = state.copy()
        new_state["messages"] = state["messages"] + [response]
        new_state["current_step"] = state["current_step"] + 1
        
        return new_state
    
    def _action_node(self, state: AgentState) -> Dict[str, Any]:
        """工具执行节点"""
        print(f"\n=== 执行工具 ===")
        
        # 获取最后一条消息
        last_message = state["messages"][-1]
        
        # 执行工具调用
        tool_outputs = []
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                print(f"调用工具: {tool_call['name']}")
                print(f"参数: {tool_call['args']}")
                
                # 执行工具
                tool_output = self.tool_executor.invoke(tool_call)
                tool_outputs.append(tool_output)
                
                print(f"工具输出: {tool_output.content[:200]}...")
                
                # 记录历史
                if tool_call['name'] == 'execute_shell_command':
                    state["shell_history"].append({
                        'command': tool_call['args'].get('command', ''),
                        'result': tool_output.content,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    state["api_calls"].append({
                        'tool': tool_call['name'],
                        'input': tool_call['args'],
                        'result': tool_output.content,
                        'timestamp': datetime.now().isoformat()
                    })
        
        # 更新状态
        new_state = state.copy()
        new_state["messages"] = state["messages"] + tool_outputs
        
        return new_state
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """判断是否继续执行"""
        last_message = state["messages"][-1]
        
        # 检查是否完成
        if hasattr(last_message, 'content') and last_message.content:
            if "FINISHED:" in last_message.content:
                print("任务完成！")
                return "end"
        
        # 检查步数限制
        if state["current_step"] >= state["max_steps"]:
            print(f"达到最大步数限制: {state['max_steps']}")
            return "end"
        
        # 检查是否有工具调用
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        
        return "end"
    
    def run(self, task: str, max_steps: int = 10) -> Dict[str, Any]:
        """运行任务"""
        print(f"开始执行任务: {task}")
        
        # 初始化状态
        initial_state: AgentState = {
            "messages": [HumanMessage(content=task)],
            "current_step": 0,
            "max_steps": max_steps,
            "task_completed": False,
            "shell_history": [],
            "api_calls": []
        }
        
        # 运行工作流
        final_state = self.graph.invoke(initial_state)
        
        # 检查任务是否完成
        last_message = final_state["messages"][-1]
        task_completed = False
        if hasattr(last_message, 'content') and last_message.content:
            task_completed = "FINISHED:" in last_message.content
        
        # 返回结果
        return {
            'task': task,
            'completed': task_completed,
            'steps': final_state["current_step"],
            'shell_history': final_state["shell_history"],
            'api_calls': final_state["api_calls"],
            'messages': final_state["messages"]
        }

# 使用示例
def main():
    """主函数 - 演示用法"""
    
    print("=== LangGraph ReAct Shell执行器演示 ===\n")
    
    # 创建代理
    agent = ReActShellAgent()
    
    # 示例任务
    tasks = [
        "获取系统信息",
        "列出当前目录的文件",
        "搜索Python相关的GitHub仓库",
        "获取北京的天气信息"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"任务 {i}: {task}")
        print(f"{'='*60}")
        
        try:
            result = agent.run(task, max_steps=5)
            
            print(f"\n📊 执行结果:")
            print(f"  - 任务完成: {result['completed']}")
            print(f"  - 执行步数: {result['steps']}")
            print(f"  - Shell命令: {len(result['shell_history'])}")
            print(f"  - API调用: {len(result['api_calls'])}")
            
            if result['shell_history']:
                print(f"\n🔧 Shell命令历史:")
                for cmd in result['shell_history']:
                    print(f"  • {cmd['command']}")
            
            if result['api_calls']:
                print(f"\n🌐 API调用历史:")
                for call in result['api_calls']:
                    print(f"  • {call['tool']}")
                    
        except Exception as e:
            print(f"❌ 任务执行错误: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*60}")
        
        # 如果不是最后一个任务，等待用户输入
        if i < len(tasks):
            input("按回车继续下一个任务...")

if __name__ == "__main__":
    # 检查是否有OpenAI API密钥
    #if OPENAI_API_KEY == "your-openai-api-key":
    #    print("⚠️  请先设置OPENAI_API_KEY")
    #    print("可以通过环境变量或直接修改代码中的OPENAI_API_KEY变量")
    #    exit(1)
    
    main()

