from typing import List, Sequence

from langchain.agents import ConversationalChatAgent, AgentExecutor, create_react_agent, Agent
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool, BaseTool
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st


class OrchestratorAgent:
    def __init__(self, tools, llm, memory: ConversationBufferMemory):
        self.tools: Sequence[BaseTool] = tools
        self.llm: ChatOpenAI = llm
        self.chat_agent: Agent = ConversationalChatAgent.from_llm_and_tools(llm=self.llm, tools=self.tools)
        self.executor: AgentExecutor = AgentExecutor.from_agent_and_tools(
            agent=self.chat_agent,
            tools=self.tools,
            memory=memory,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

    def get_streamlit_response(self, container, msgs, prompt):
        st_cb = StreamlitCallbackHandler(container, expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]
        response = self.executor.invoke(prompt, cfg)
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]
        st.write(response["output"])