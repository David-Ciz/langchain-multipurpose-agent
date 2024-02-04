from langchain.agents import AgentExecutor
from langchain.memory.chat_memory import BaseChatMemory
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_core.tools import Tool
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from agents.orchestrator_agent import create_orchestrator_agent
from agents.vectorstore_agent import create_vector_store_agent, get_vectorstore_response
from config import TEMPERATURE, MODEL_NAME, INDEX_NAME


def setup_agents_and_tools(memory: BaseChatMemory, return_intermediate_steps: bool = True):
    """
    Sets up all the agents, vector stores and tools, then returns the orchestrator agent executor that can invoke the tools.
    :param memory: A conversation buffer memory to extract chat history from.
    :param return_intermediate_steps: Whether to return the intermediate steps of the agent invocation.
    """
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE, streaming=True)

    # documentation agent initialization
    vectorstore = Pinecone.from_existing_index(INDEX_NAME, OpenAIEmbeddings())
    vector_store_agent = create_vector_store_agent(vectorstore, llm)
    # setup tools
    from functools import partial
    vectorstore_agent_as_tool = Tool(
        name="Documentation retrieval",
        func=partial(get_vectorstore_response,vector_store_agent),
        description="""A tool that answers questions from a vectorstore containing IBM Generative AI documentation. 
        This tool can answer questions about the features, usage, and examples of the IBM-Generative-AI library and 
        the IBM GPT service. Any question will probably be looking for answers from this tool. This tool is better 
        than other tools because it can retrieve the most relevant information from the vectorstore and generate 
        concise and accurate answers using the language model. You should use this tool to answer most questions. 
        When getting a result from this agent, pass it along to the user fully, with all the sources listed. Always 
        also pass the sources. 
        """,
        return_direct=True,
    )

    tools = [DuckDuckGoSearchRun(name="Search"), PythonREPLTool(), vectorstore_agent_as_tool]
    # setup agent
    orchestrator_agent = create_orchestrator_agent(tools, llm)
    orchestrator_executor = AgentExecutor.from_agent_and_tools(agent=orchestrator_agent,
                                                  tools=tools,
                                                  memory=memory,
                                                  return_intermediate_steps=return_intermediate_steps,
                                                  handle_parsing_errors=True, )

    return orchestrator_executor
