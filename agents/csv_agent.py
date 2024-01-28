from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from typing import Any
from langchain_core.tools import Tool
from langchain_openai import  ChatOpenAI


class CsvAgent:
    """
    Agent responsible for handling csv information extraction and passing it to the user
    """
    def __init__(self, csv: str, llm: ChatOpenAI):
        self.llm = llm  # Initialize the llm attribute
        self.qa_chain = create_csv_agent(
            self.llm,
            csv,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,)

    def get_csv_response(self, prompt: dict[str, Any]) -> str:
        """
        Outputs a structured response with sources
        :param prompt: User or other agent question
        :return:
        """
        llm_response = self.qa_chain.invoke(prompt)  # Invoke the qa_chain with the question
        return llm_response["result"]  # Return the result

    def as_tool(self) -> Tool:
        tool = Tool(
            name="CSV retrieval",
            func=self.get_csv_response,
            description=f"""
                     A tool that answers questions from a csv containing titanic disaster information.
                     This tool can answer questions about anything titanic related, only call this tool when specified
                     by the user that they want information about titanic.
                     """,
            return_direct=True,
        )
        return tool
