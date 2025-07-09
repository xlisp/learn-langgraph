"""
LangGraph ReAct Shell执行器 - 结合第三方API
支持推理、行动、观察循环，执行shell命令和API调用
"""

import json
import subprocess
import requests
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import os
from datetime import datetime

# 配置
OPENAI_API_KEY = "your-openai-api-key"  # 请替换为您的API密钥
WEATHER_API_KEY = "your-weather-api-key"  # 请替换为您的天气API密钥

@dataclass
class AgentState:
    """Agent状态管理"""
    messages: List[Dict[str, Any]]
    current_step: int
    max_steps: int
    task_completed: bool
    shell_history: List[Dict[str, str]]
    api_calls: List[Dict[str, Any]]
    
    def __init__(self):
        self.messages = []
        self.current_step = 0
        self.max_steps = 10
        self.task_completed = False
        self.shell_history = []
        self.api_calls = []

# 工具定义
@tool
def execute_shell_command(command: str, timeout: int = 30) -> str:
    """
    执行shell命令
    
    Args:
        command: 要执行的shell命令
        timeout: 超时时间（秒）
    
    Returns:
        命令执行结果
    """
    try:
        # 安全检查 - 避免危险命令
        dangerous_commands = ['rm -rf', 'sudo', 'chmod +x', 'wget', 'curl -X DELETE']
        if any(dangerous in command.lower() for dangerous in dangerous_commands):
            return f"错误: 命令 '{command}' 被安全策略阻止"
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = f"退出码: {result.returncode}\n"
        output += f"标准输出:\n{result.stdout}\n"
        if result.stderr:
            output += f"标准错误:\n{result.stderr}\n"
        
        return output
    
    except subprocess.TimeoutExpired:
        return f"错误: 命令执行超时 ({timeout}秒)"
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
        # 使用OpenWeatherMap API (需要注册获取API密钥)
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city,
            'appid': WEATHER_API_KEY,
            'units': 'metric',
            'lang': 'zh'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            weather_info = {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'description': data['weather'][0]['description'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure']
            }
            return json.dumps(weather_info, ensure_ascii=False, indent=2)
        else:
            return f"错误: 无法获取天气信息，状态码: {response.status_code}"
    
    except Exception as e:
        return f"错误: {str(e)}"

@tool
def search_github_repos(query: str, language: str = "") -> str:
    """
    搜索GitHub仓库
    
    Args:
        query: 搜索关键词
        language: 编程语言过滤（可选）
    
    Returns:
        搜索结果JSON字符串
    """
    try:
        url = "https://api.github.com/search/repositories"
        params = {
            'q': query + (f" language:{language}" if language else ""),
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
        import psutil
        
        info = {
            'system': platform.system(),
            'node': platform.node(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            'memory_available': f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
            'disk_usage': f"{psutil.disk_usage('/').percent}%"
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
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """创建LangGraph工作流"""
        graph = StateGraph(AgentState)
        
        # 添加节点
        graph.add_node("reasoner", self._reasoning_node)
        graph.add_node("tool_executor", self._tool_execution_node)
        graph.add_node("observer", self._observation_node)
        
        # 添加边
        graph.add_edge("reasoner", "tool_executor")
        graph.add_edge("tool_executor", "observer")
        graph.add_conditional_edges(
            "observer",
            self._should_continue,
            {
                "continue": "reasoner",
                "end": END
            }
        )
        
        # 设置入口点
        graph.set_entry_point("reasoner")
        
        return graph.compile()
    
    def _reasoning_node(self, state: AgentState) -> Dict[str, Any]:
        """推理节点 - 分析当前状态并决定下一步行动"""
        print(f"\n=== 推理步骤 {state.current_step + 1} ===")
        
        # 构建系统提示
        system_prompt = """你是一个智能助手，可以执行shell命令和调用API。
        
可用工具:
1. execute_shell_command - 执行shell命令
2. get_weather_info - 获取天气信息
3. search_github_repos - 搜索GitHub仓库
4. get_system_info - 获取系统信息

请按照ReAct模式工作:
1. Thought (思考): 分析当前情况，决定需要做什么
2. Action (行动): 选择合适的工具和参数
3. Observation (观察): 分析执行结果

请用JSON格式回复:
{
    "thought": "你的思考过程",
    "action": "工具名称",
    "action_input": "工具参数"
}

如果任务完成，请回复:
{
    "thought": "任务完成的总结",
    "action": "finish",
    "action_input": "最终结果"
}"""
        
        # 构建对话历史
        messages = [SystemMessage(content=system_prompt)]
        
        # 添加用户消息
        if state.messages:
            messages.extend([
                HumanMessage(content=msg['content']) 
                for msg in state.messages if msg['role'] == 'user'
            ])
        
        # 添加执行历史
        if state.shell_history:
            history = "执行历史:\n"
            for cmd in state.shell_history[-3:]:  # 只显示最近3条
                history += f"命令: {cmd['command']}\n结果: {cmd['result'][:200]}...\n\n"
            messages.append(HumanMessage(content=history))
        
        # 获取LLM响应
        response = self.llm.invoke(messages)
        
        try:
            # 解析JSON响应
            decision = json.loads(response.content)
            print(f"思考: {decision['thought']}")
            print(f"行动: {decision['action']}")
            print(f"参数: {decision['action_input']}")
            
            state.messages.append({
                'role': 'assistant',
                'content': response.content,
                'decision': decision
            })
            
        except json.JSONDecodeError:
            print("JSON解析错误，使用默认决策")
            decision = {
                'thought': '无法解析响应，获取系统信息',
                'action': 'get_system_info',
                'action_input': ''
            }
            state.messages.append({
                'role': 'assistant',
                'content': json.dumps(decision, ensure_ascii=False),
                'decision': decision
            })
        
        return {"state": state}
    
    def _tool_execution_node(self, state: AgentState) -> Dict[str, Any]:
        """工具执行节点"""
        print(f"\n=== 执行工具 ===")
        
        if not state.messages:
            return {"state": state}
        
        last_message = state.messages[-1]
        decision = last_message.get('decision', {})
        
        action = decision.get('action', '')
        action_input = decision.get('action_input', '')
        
        # 检查是否完成
        if action == 'finish':
            state.task_completed = True
            print(f"任务完成: {action_input}")
            return {"state": state}
        
        # 执行工具
        if action in self.tools_by_name:
            tool = self.tools_by_name[action]
            try:
                if isinstance(action_input, dict):
                    result = tool.invoke(action_input)
                else:
                    result = tool.invoke(action_input) if action_input else tool.invoke("")
                
                print(f"工具执行结果: {result[:300]}...")
                
                # 记录执行历史
                if action == 'execute_shell_command':
                    state.shell_history.append({
                        'command': action_input,
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    state.api_calls.append({
                        'tool': action,
                        'input': action_input,
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    })
                
                state.messages.append({
                    'role': 'tool',
                    'content': result,
                    'tool_name': action
                })
                
            except Exception as e:
                error_msg = f"工具执行错误: {str(e)}"
                print(error_msg)
                state.messages.append({
                    'role': 'tool',
                    'content': error_msg,
                    'tool_name': action
                })
        else:
            error_msg = f"未知工具: {action}"
            print(error_msg)
            state.messages.append({
                'role': 'tool',
                'content': error_msg,
                'tool_name': action
            })
        
        return {"state": state}
    
    def _observation_node(self, state: AgentState) -> Dict[str, Any]:
        """观察节点 - 分析执行结果"""
        print(f"\n=== 观察结果 ===")
        
        if state.messages:
            last_message = state.messages[-1]
            if last_message.get('role') == 'tool':
                print(f"工具输出: {last_message['content'][:200]}...")
        
        state.current_step += 1
        
        return {"state": state}
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """判断是否继续执行"""
        if state.task_completed:
            return "end"
        
        if state.current_step >= state.max_steps:
            print(f"达到最大步数限制: {state.max_steps}")
            return "end"
        
        return "continue"
    
    def run(self, task: str) -> Dict[str, Any]:
        """运行任务"""
        print(f"开始执行任务: {task}")
        
        # 初始化状态
        state = AgentState()
        state.messages.append({
            'role': 'user',
            'content': task
        })
        
        # 运行图
        final_state = self.graph.invoke(state)
        
        # 返回执行结果
        return {
            'task': task,
            'completed': final_state.task_completed,
            'steps': final_state.current_step,
            'shell_history': final_state.shell_history,
            'api_calls': final_state.api_calls,
            'messages': final_state.messages
        }

# 使用示例
def main():
    """主函数 - 演示用法"""
    
    # 创建代理
    agent = ReActShellAgent()
    
    # 示例任务
    tasks = [
        "检查当前系统信息，然后查看当前目录下的文件",
        "搜索Python相关的热门GitHub仓库",
        "获取北京的天气信息",
        "列出当前目录的内容，然后创建一个test文件夹"
    ]
    
    print("=== LangGraph ReAct Shell执行器演示 ===\n")
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*50}")
        print(f"任务 {i}: {task}")
        print(f"{'='*50}")
        
        try:
            result = agent.run(task)
            
            print(f"\n任务完成状态: {result['completed']}")
            print(f"执行步数: {result['steps']}")
            print(f"Shell命令数: {len(result['shell_history'])}")
            print(f"API调用数: {len(result['api_calls'])}")
            
            if result['shell_history']:
                print("\nShell执行历史:")
                for cmd in result['shell_history']:
                    print(f"  - {cmd['command']}")
            
            if result['api_calls']:
                print("\nAPI调用历史:")
                for call in result['api_calls']:
                    print(f"  - {call['tool']}: {str(call['input'])[:50]}...")
                    
        except Exception as e:
            print(f"任务执行错误: {e}")
        
        print("\n" + "="*50)
        input("按回车继续下一个任务...")

if __name__ == "__main__":
    main()
