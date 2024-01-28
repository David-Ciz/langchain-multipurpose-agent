from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st

import logging
from utils import StreamHandler, process_source_metadata

logger = logging.getLogger(__name__)

# Could be made with classes, But I'm still not losing hope I can combine them in some clever way and they have different parameter amounts anyway, this is more readable for now.
# ------------------DOCUMENTATION AGENT --------------------


def initiate_documentation_agent(INDEX_NAME, chat_box, memory):
    # vector store initiation for the database qa with sources
    try:
        vectorstore = Pinecone.from_existing_index(INDEX_NAME, OpenAIEmbeddings())
        # TODO: add reranking to the chain
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    except Exception as e:
        logger.error(f"An error occurred while connecting to Pinecone: {e}")
    else:
        logger.info("Successfully connected to Pinecone")

    stream_handler = StreamHandler(chat_box, display_method='write')
    template = """You are an AI chatbot having a conversation with a human. Use primarily the information from
    the summaries. If you don't know the answer, it's better to say 'I'm not sure' rather than make stuff up.
    {summaries}
    {chat_history}
    Human: {question}
    AI: """
    prompt_template = PromptTemplate(input_variables=["summaries", "question"], template=template)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, streaming=True, callbacks=[stream_handler])
    chain_type_kwargs = {"prompt": prompt_template}
    # TODO: check the summaries, I don't think they are working correctly right now.
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever=retriever,
                                                           memory=memory, return_source_documents=True,
                                                           chain_type_kwargs=chain_type_kwargs)
    return qa_chain


def get_documentation_qa_response(executor, chat_box, prompt):
    for response in executor.stream({"question": prompt}):
        processed_response = process_source_metadata(response)
        chat_box.write(processed_response)

# ------------------TOOL AGENT --------------------


# def initiate_tools_agent(memory):
#     tools = [DuckDuckGoSearchRun(name="Search"),
#              PythonREPLTool()]
#     # TODO: streaming callback is just ugly for the agent, would have to parse it for the final_output,
#     #  no time unfortunately so 'streaming' is just for the inbetween steps
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, streaming=True)
#     chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
#     executor = AgentExecutor.from_agent_and_tools(
#         agent=chat_agent,
#         tools=tools,
#         memory=memory,
#         return_intermediate_steps=True,
#         handle_parsing_errors=True,
#     )
#     return executor
#
#
# def get_tools_response(executor, container, msgs, prompt):
#     st_cb = StreamlitCallbackHandler(container, expand_new_thoughts=False)
#     cfg = RunnableConfig()
#     cfg["callbacks"] = [st_cb]
#     response = executor.invoke(prompt, cfg)
#     st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]
#     st.write(response["output"])