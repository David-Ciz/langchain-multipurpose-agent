from typing import Sequence

from langchain.agents import create_react_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from langchain_core.prompts import ChatPromptTemplate

from agents.prompts.prompt_templates import orchestrator_prompt


def create_orchestrator_agent(tools: Sequence[BaseTool],
                              llm: BaseLanguageModel,
                              prompt_template: str = orchestrator_prompt(),
                              ) -> Runnable:
    """
    Creates a main agent that calls on other tools and agents.
    :param tools: A sequence of tools for the agent to use.
    :param llm: A language model that will be used for the executor invocations.

    :param prompt_template: A prompt template to use for the agent, defaults to orchestrator_prompt defined in prompts.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chat_agent: Runnable = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return chat_agent
