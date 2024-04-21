"""
Large Language Model College Football Data: Server
Author: Trevor Cross
Last Updated: 04/19/24

Build and serve langchain agent to interact w/ BigQuery database and answer questions.
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
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
    )
from langchain.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.agent_toolkits import create_sql_agent

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

# create system prefix
sys_prefix = """
You are an agent designed to interact with Google BigQuery SQL database containing data on college football. Given an input question, create a syntactically correct GoogleSQL query to run, then look at the results of the query, and return the query and answer. You have access to tools for interacting with the databse. Only use the given tools. Only use the information returned by the tools to construct your final answer. You must obtain schema information on all available tables before writing SQL queries. You must double check your query before executing it, making sure you query existing tables and fields. If you get an error while executing a query, rewrite the query and try again.

Do not make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the databse.

If the question does not seem to be related to the database and college football, don't return anything for the SQL query and only return "I'm unable to answer that." as the answer.

Make sure to list all available table info with a couple rows using GoogleSQL before executing any other queries. You may have to use subqueries to get your final answer.

The following are examples of input questions, and output SQL queries and answers:
"""

# define examples
example_query = """SELECT COUNT(*) as total_wins FROM `llm-cfd.raw.game_data` WHERE (home_team="Wisconsin" AND home_points>away_points AND season=2017) OR (away_team="Wisconsin" AND home_points<away_points AND season=2017)"""

examples = [
    {
        "input": "How many games did Wisconsin win in 2017?",
        "sql_query": example_query,
        "answer": f"Wisconsin won 13 games in the 2017 season. This is the query I used: \n{example_query}"
    }
]

# create example template
example_template = """Input: {input}
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
    prefix=sys_prefix,
    suffix="",
    input_variables=["input"],
    )

# create final prompt template
full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("user", "{input}"),
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

#llm = Ollama(
#    base_url='http://localhost:11434',
#    model="duckdb-nsql"
#    )

# initialize LLM agent
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    prompt=full_prompt,
    verbose=True,
    agent_type="openai-tools",
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
