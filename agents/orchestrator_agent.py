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

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
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
        #self.prompt: PromptTemplate = hub.pull(prompt_address)
        self.prompt = ChatPromptTemplate.from_template(custom_prompt())
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


def custom_prompt() :
    prompt = """Assistant is a large language model trained by OpenAI.

                Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
                
                Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
                
                Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
                
                TOOLS:
                ------
                
                Assistant has access to the following tools:
                
                {tools}
                
                To use a tool, please use the following format:
                
                ```
                Thought: Do I need to use a tool? Yes
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ```
                
                When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
                
                ```
                Thought: Do I need to use a tool? No
                Final Answer: [your response here]
                ```
                
                Begin! Remember to always give a COMPLETE answer e.g. after a "Thought:" with  "Do i need to use a tool? Yes/No" follows ALWAYS in a new line Action: (...) or Final Answer: (...), as described above.
                Previous conversation history:
                {chat_history}
                
                New input: {input}
                {agent_scratchpad}"""
    return prompt
