"""
Large Language Model College Football Data: Server
Author: Trevor Cross
Last Updated: 04/24/24

Build and serve langchain agent to interact w/ BigQuery database and answer questions.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import langchain libraries
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
    )
from langchain.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_sql_agent

# import server libraries
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

# import support libraries
import sys
import os

# import toolbox functions
sys.path.append("src")
from toolbox import *

# -----------------------
# ---Input Credentials---
# -----------------------

# input path to BQ service accout key
key_path = os.path.join(".secrets", "llm-cfd-4a97db654318.json")

# input BQ identifiers
project = "llm-cfd"
dataset = "raw"
table = "game_data"

# input sqlalchemy path
sqlalchemy_url = f'bigquery://{project}/{dataset}?credentials_path={key_path}'

# open OpenAI API key
openai_key_path = os.path.join(".secrets", "openai_key.json")
openai_key = json_to_dict(openai_key_path)['api_key']

# ----------------------------
# ---Define Prompt Template---
# ----------------------------

# pull system prefix
prefix_path = os.path.join("prompt_files", "system_prefix.txt")
with open(prefix_path, 'r') as file:
    prefix = file.read()

# define system suffix
suffix = "Here's the chat history:"

# pull examples
example_path = os.path.join("prompt_files", "examples.json")
examples = json_to_dict(example_path)

# create example template
example_template = """input: {input}
sql_query: {sql_query}
answer: {answer}
"""

# create example prompt template
example_prompt = PromptTemplate(
    input_variables=["input", "sql_query", "answer"],
    template=example_template
    )

# create few shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["chat_history", "input"],
    )

# create final prompt template
full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ]
)

# ---------------------------
# ---Build LangChain Agent---
# ---------------------------

# initialize conversational memory object
memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    return_messages=True
    )

# initialize SQL DB
db = SQLDatabase.from_uri(sqlalchemy_url)

# initialize chat LLM
chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    openai_api_key=openai_key
    )

#chat = ChatOllama(
#    base_url='http://localhost:11434',
#    model="duckdb-nsql"
#    )

# create OpenAI tools agent w/ SQL DB tools
agent = create_sql_agent(
    llm=chat,
    db=db,
    prompt=full_prompt,
    agent_type='zero-shot-react-description',
    agent_executor_kwargs={"memory": memory},
    verbose=True
    )

# ---------------------------
# ---Define App Deployment---
# ---------------------------

# define app
app = FastAPI(
    title="LangChain Server",
    description="An API server meant to query college football data in Google BigQuery",
    )

# define root response
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# define primary (SQL DB) response
add_routes(
    app,
    agent,
    path="/CFD",
    )

# run app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
