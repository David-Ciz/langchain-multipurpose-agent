from typing import Sequence

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks import FinalStreamingStdOutCallbackHandler
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.tools import BaseTool

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

# import abc
#
# @abc.ABC
# class BaseAgent:
#     def __init__(self, tools: Sequence[BaseTool], llm: ChatOpenAI, memory: ConversationBufferMemory,
#                  return_intermediate_steps, prompt_address="hwchase17/react-chat"):
#         ...
#     @abc.abstractmethod
#     def invoke(self, prompt):
#         raise NotImplemented
#
#
# class StreamlitAgent(BaseAgent):
#     def __init__(self, tools: Sequence[BaseTool], llm: ChatOpenAI, memory: ConversationBufferMemory,
#                  container: DeltaGenerator, msgs: StreamlitChatMessageHistory, return_intermediate_steps, prompt_address="hwchase17/react-chat"):
#
#
# # def init_executor(....)
#
#
# def call_executor(executor, prompt, ..., handler = None):
#     if handler is not None:
#         cfg = RunnableConfig()
#         cfg["callbacks"] = [handler]
#     executor.in...

class OrchestratorAgent:
    """
    Main agent that calls on other tools and agents.
    """
    def __init__(self, tools: Sequence[BaseTool], llm: BaseLanguageModel, memory: ConversationBufferMemory,
                 return_intermediate_steps, prompt_address="hwchase17/react-chat"):
        self.tools: Sequence[BaseTool] = tools
        self.llm: ChatOpenAI = llm
        self.prompt: PromptTemplate = hub.pull(prompt_address)
        self.chat_agent: Runnable = create_react_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        self.executor: AgentExecutor = AgentExecutor.from_agent_and_tools(
            agent=self.chat_agent,
            tools=self.tools,
            memory=memory,
            return_intermediate_steps=return_intermediate_steps,
            handle_parsing_errors=True,
        )

    # @staticmethod
    # def create_standard_agent(llm: BaseChatModel):
    #     ...
    #

    def invoke_streamlit_response(self, container: DeltaGenerator, msgs: StreamlitChatMessageHistory, prompt: str):
        """
        Returns a streaming response to the streamlit chat
        :param container: container to write response into
        :param msgs: streamlit history handler
        :param prompt: user input prompt
        :return:
        """
        # callback to see the thoughts of the agent
        st_cb = StreamlitCallbackHandler(container, expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]
        # You must return a final answer, not an action. If you can't return a final answer, return "none".
        # https://github.com/langchain-ai/langchain/issues/1358
        response = self.executor.invoke({"input": prompt, "chat_history": msgs}, cfg)
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]
        st.write(response["output"])

    def get_cli_response(self, prompt: str, verbose: bool):
        """
        Returns a streaming response to the cli
        :param prompt: Question or request from the user
        :param verbose: If the whole thought chain should be streamed or not
        :return:
        """

        cfg = RunnableConfig()
        if verbose:
            handler = StreamingStdOutCallbackHandler()
            cfg["callbacks"] = [handler]
            self.executor.invoke({"input": prompt, "chat_history": []}, cfg)
        else:
            handler = FinalStreamingStdOutCallbackHandler()
            cfg["callbacks"] = [handler]
            self.executor.invoke({"input": prompt, "chat_history": []}, cfg)
        # there shouldn't be a need to return anything, since the result gets streamed straight to cli.

    def get_api_response(self, prompt: str):
        result = self.executor.invoke({"input": prompt, "chat_history": []})
        return result
