import logging
import os

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config
from agents.csv_agent import CsvAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.output_parsers.parsers import parse_agent_messages
from agents.vectorstore_agent import VectorStoreAgent
from config import TEMPERATURE
from styles import css

# env setup, make sure you have a .env file under root with
# OPENAI_API_KEY
# PINECONE_API_KEY
# PINECONE_ENVIRONMENT
# INDEX_NAME

logger = logging.getLogger("Assistant")
missing_variables = config.load_env()
if missing_variables:
    st.warning(f'warning, you are missing the following env. variables: {missing_variables}', icon="⚠️")

logger = logging.getLogger(__name__)
INDEX_NAME = os.environ["INDEX_NAME"]

with st.sidebar:
    # Since the database search doesn't play nice with the other tools, I seperate it into two.
    # agent_type = st.radio("select tools", ["database_search_agent", "tool_agent"])
    # if agent_type == "tool_agent":
    uploaded_file = st.file_uploader("You can upload files! (didn't make the csv tool in time though :( )")
    st.markdown(css, unsafe_allow_html=True)

welcome_ai_message = " Hello, I'm a helpful assistant that can answer questions from the documentation. " \
                     "I can search internet for you and execute python code!"
llm = ChatOpenAI(model_name=config.MODEL_NAME, temperature=TEMPERATURE, streaming=True)

# documentation agent initialization
vectorstore = Pinecone.from_existing_index(INDEX_NAME, OpenAIEmbeddings())
documentation_agent = VectorStoreAgent(llm=llm, vectorstore=vectorstore)

# csv agent initialization
csv_agent = CsvAgent("input_files/titanic.csv", llm)

# memory setup with resetting button.
msgs = StreamlitChatMessageHistory(key="special_app_key")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, chat_memory=msgs,
                                  output_key="output")
# resetting memory handler
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message(welcome_ai_message)
    st.session_state.steps = {}

parse_agent_messages(msgs)

# setup tools
tools = [DuckDuckGoSearchRun(name="Search"), PythonREPLTool(), documentation_agent.as_tool()]

# setup agent
orchestrator_agent = OrchestratorAgent(tools, llm, memory, return_intermediate_steps=True)

# chat interface
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        # try:
        container = st.container()
        try:
            orchestrator_agent.invoke_streamlit_response(container, msgs, prompt)
        except Exception as e:
            # Very basic exception handling, mostly here so that the user doesn't get some ugly response.
            # TODO: Maybe add something here to disable the chat_box? details, and solutions are ugly.
            logger.error(f"An error occurred while generating answer: {e}")
            container.write("Something went wrong, please try rerunning the program and look at "
                            "logger errors")
