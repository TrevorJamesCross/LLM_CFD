"""
Large Language Model College Football Data: Server
Author: Trevor Cross
Last Updated: 04/16/24

Build and server langchain agent w/ SQL toolkit & conversational memory.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import server libraries
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

# import langchain libraries
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate
    )
from langchain_community.agent_toolkits import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

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

# define system prefix
sys_message = """You are an agent designed to interact with Google BigQuery SQL database. Given an input question, create a syntactically correct GoogleSQL query to run, then look at the results of the query and return the answer. You have access to tools for interacting with the databse. Only use the given tools. Only use the information returned by the tools to construct your final answer. You must double check your query before executing it, making sure you query existing tables and fields. If you get an error while executing a query, rewrite the query and try again.

Do not make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the databse.

If the question does not seem to be related to the databse, just return "I'm unable to answer that." as the answer."""

# create prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", sys_message),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ]
)

# ---------------------------
# ---Build LangChain Agent---
# ---------------------------

# initialize SQL DB
db = SQLDatabase.from_uri(sqlalchemy_url)

# initialize LLM
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    openai_api_key=openai_key
    )

# initialize LLM agent
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    #prompt=prompt_template,
    verbose=True,
    top_k=1000,
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
    agent_executor,
    path="/CFD",
    )

# run app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
