import logging
import os

from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain.callbacks import StdOutCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig
from langchain_experimental.tools.python.tool import PythonREPLTool

from agents import initiate_documentation_agent, initiate_tools_agent, get_documentation_qa_response, get_tools_response
from styles import css
from utils import StreamHandler, process_source_metadata, parse_agent_messages

# env setup, make sure you have a .env file under root with
# OPENAI_API_KEY
# PINECONE_API_KEY
# PINECONE_ENVIRONMENT
# INDEX_NAME
load_dotenv()
logger = logging.getLogger(__name__)
INDEX_NAME = os.environ["INDEX_NAME"]

with st.sidebar:
    # Since the database search doesn't play nice with the other tools, I seperate it into two.
    agent_type = st.radio("select tools", ["database_search_agent", "tool_agent"])
    if agent_type == "tool_agent":
        uploaded_file = st.file_uploader("You can upload files!")
        st.markdown(css, unsafe_allow_html=True)
        # each agent has different memory output key
        memory_output_key = "output"
    else:
        memory_output_key = "answer"


# handler = StdOutCallbackHandler() # doesn't work with the stream callback, can be good for debugging though.

# memory setup with resetting button.
msgs = StreamlitChatMessageHistory(key="special_app_key")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, chat_memory=msgs, output_key=memory_output_key)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}


if agent_type == "tool_agent":
    parse_agent_messages(msgs)
else:
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        if agent_type == "database_search_agent":
            chat_box = st.empty()
            executor = initiate_documentation_agent(INDEX_NAME, chat_box, memory)
        else:
            executor = initiate_tools_agent(memory)
        try:
            if agent_type == "database_search_agent":
                get_documentation_qa_response(executor, chat_box, prompt)
            else:
                container = st.container()
                get_tools_response(executor,container, msgs, prompt)
        except Exception as e:
            # TODO: Maybe add something here to disable the chat_box? details, and solutions are ugly.
            logger.error(f"An error occurred while generating answer: {e}")
            chat_box.write("Something went wrong, please try rerunning the program and look at logger errors")

