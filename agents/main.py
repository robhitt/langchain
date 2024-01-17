from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

load_dotenv()

handler = ChatModelStartHandler()
chat = ChatOpenAI(callbacks=[handler])


tables = list_tables()

# agent_scratchpad holds the state of the agent. think- memory for talking back and forth
# goal of it is to capture intermediate messages.
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content=(
                "You are an ai that has access to a SQLite database.\n"
                f"The database has tables of: {tables}\n"
                "Do not make any assumptions about what tables exist "
                "or what columns exist. Instead, use the 'describe_tables' function"
            )
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools = [run_query_tool, describe_tables_tool, write_report_tool]

# An agent is almost identical to a chain but it knows how to use tools.
agent = OpenAIFunctionsAgent(llm=chat, prompt=prompt, tools=tools)

# Fancy while loop that lets you chat with the agent.
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)  # verbose=True,

# agent_executor("How many users have provided a shipping address?")
agent_executor("How many orders are there? Write the results to a report file.")

agent_executor("Repeat the exact same process for users.")
