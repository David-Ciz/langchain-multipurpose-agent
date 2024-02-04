from typing import Any

from langchain.agents import AgentExecutor
from langchain.callbacks import FinalStreamingStdOutCallbackHandler
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnableConfig
import streamlit as st
from langchain_core.tools import Tool
from streamlit.delta_generator import DeltaGenerator


def invoke_streamlit_response(executor: AgentExecutor, container: DeltaGenerator, msgs: StreamlitChatMessageHistory,
                              prompt: str):
    """
    Returns a streaming response to the streamlit chat
    :param executor: An instance of the agent executor for invocation
    :param container: container to write response into
    :param msgs: streamlit history handler
    :param prompt: Question or request from the user
    """
    # callback to see the thoughts of the agent
    st_cb = StreamlitCallbackHandler(container, expand_new_thoughts=False)
    cfg = RunnableConfig()
    cfg["callbacks"] = [st_cb]
    response = executor.invoke({"input": prompt, "chat_history": msgs}, cfg)
    st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]
    st.write(response["output"])


def invoke_cli_response(executor: AgentExecutor, prompt: str, verbose: bool):
    """
    Returns a streaming response to the cli directly
    :param executor: An instance of the agent executor for invocation
    :param prompt: Question or request from the user
    :param verbose: If the whole thought chain should be streamed or not
    """

    cfg = RunnableConfig()
    if verbose:
        handler = StreamingStdOutCallbackHandler()
    else:
        handler = FinalStreamingStdOutCallbackHandler()
    cfg["callbacks"] = [handler]
    executor.invoke({"input": prompt, "chat_history": []}, cfg)
    # there shouldn't be a need to return anything, since the result gets streamed straight to cli.


def get_api_response(executor: AgentExecutor, prompt: str) -> dict[str, Any]:
    """
    Returns a response from the agent for api invocation
    :param executor: An instance of the agent executor for invocation
    :param prompt: Question or request from the user
    """
    result = executor.invoke({"input": prompt, "chat_history": []})
    return result

def get_csv_response(csv_agent, prompt: dict[str, Any]) -> str:
    """
    Outputs a structured response with sources for a csv agent
    :param prompt: User or other agent question
    :return:
    """
    llm_response = csv_agent.invoke(prompt)
    return llm_response["result"]


def agent_as_tool(agent, invocation_function, description, ) -> Tool:
    tool = Tool(
        name="CSV retrieval",
        func=get_csv_response,
        description=f"""
                 A tool that answers questions from a csv containing titanic disaster information.
                 This tool can answer questions about anything titanic related, only call this tool when specified
                 by the user that they want information about titanic.
                 """,
        return_direct=True,
    )
    return tool





