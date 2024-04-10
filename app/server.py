"""
Large Language Model College Football Data: Server
Author: Trevor Cross
Last Updated: 03/25/24

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
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_openai import OpenAI
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

# ---------------------------
# ---Build LangChain Agent---
# ---------------------------

# initialize SQL DB
db = SQLDatabase.from_uri(sqlalchemy_url)

# initialize LLM
llm = OpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_key)

# initialize SQL toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# initialize LLM agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
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
