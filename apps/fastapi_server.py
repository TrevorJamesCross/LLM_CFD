"""
Large Language Model College Football Data: FastAPI Server
Author: Trevor Cross
Last Updated: 05/09/24

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
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory
from langchain.agents import create_sql_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory

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
ephemeral_chat_history = ChatMessageHistory()

# initialize SQL DB
db = SQLDatabase.from_uri(sqlalchemy_url)

# initialize chat LLM
if False:
    chat = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo",
        openai_api_key=openai_key
        )
else:
    chat = ChatOllama(
        model="llama3",
        temperature=0,
        base_url="http://localhost:11434"
        )

# create SQL agent
agent = create_sql_agent(
    llm=chat,
    db=db,
    prompt=full_prompt,
    agent_type='openai-tools',
    verbose=True
    )

# create runnable to manage agent chat history
agent_with_memory = RunnableWithMessageHistory(
    agent,
    lambda session_id: ephemeral_chat_history,
    input_messages_key="input",
    output_messages_key="output",
    history_messages_key="chat_history"
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
    agent_with_memory,
    path="/CFD",
    )

# run app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
