from langchain import hub
from langchain_community.llms.openai import OpenAI
from langchain.agents import load_tools
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description

# Load environment variables using python-dotenv
from dotenv import load_dotenv
load_dotenv()

# Prepare the large language model: OpenAI is needed here, can conveniently stop reasoning as needed
llm = OpenAI()
llm_with_stop = llm.bind(stop=["\nObservation"]) # There is an observer here to decide whether to stop.

# Prepare our tools: using DuckDuckGo search engine and a calculator based on LLM => equivalent to using OpenAI's function calling
tools = load_tools(["ddg-search", "llm-math"], llm=llm) ## Need to use ddg-search to find the weather temperatures of two cities, then use the calculator tool to calculate the temperature difference.

# Prepare the core prompt: loading the ReAct pattern prompt from LangChain Hub and filling in the tool's text description => Reflection type: reason action.
prompt = hub.pull("hwchase17/react")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# Build the Agent's workflow: the most important thing here is to save the structure of the intermediate steps into the prompt's agent_scratchpad
agent = (
    {
        "input": lambda x: x["input"],
        ## Intermediate steps, agent step design is crucial.
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser() ## Reflection LLM result parsing OutputParser
)

# Build the Agent Executor: responsible for executing the Agent workflow until the final answer (identifier) is obtained and output the answer
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "What is the temperature difference between Shanghai and Beijing today?"}) #=>  {'input': 'What is the temperature difference between Shanghai and Beijing today?', 'output': '8 degrees'}

