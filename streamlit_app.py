import logging
import os

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st

import config
from agent_setup import setup_agents_and_tools
from agents.output_parsers.parsers import parse_agent_messages
from agents.utils import invoke_streamlit_response
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
with st.sidebar:
    uploaded_file = st.file_uploader("You can upload files! currently only works with csv files)", type="csv")
    if uploaded_file:
        with open(config.UPLOAD_DATA_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Saved File")

    st.markdown(css, unsafe_allow_html=True)

welcome_ai_message = " Hello, I'm a helpful assistant that can answer questions from the documentation. " \
                     "I can search internet for you and execute python code!"

# memory setup with resetting button.
msgs = StreamlitChatMessageHistory(key="special_app_key")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, chat_memory=msgs,
                                  output_key="output")


orchestrator_agent_executor = setup_agents_and_tools(memory)

# resetting memory handler
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message(welcome_ai_message)
    st.session_state.steps = {}

parse_agent_messages(msgs)

# chat interface
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        # try:
        container = st.container()
        try:
            invoke_streamlit_response(orchestrator_agent_executor, container, msgs, prompt)
        except Exception as e:
            # Very basic exception handling, mostly here so that the user doesn't get some ugly response.
            # TODO: Maybe add something here to disable the chat_box? details, and solutions are ugly.
            logger.error(f"An error occurred while generating answer: {e}")
            container.write("Something went wrong, please try rerunning the program and look at "
                            "logger errors")
