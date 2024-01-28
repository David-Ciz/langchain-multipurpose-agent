import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor, create_json_chat_agent, \
    ConversationalChatAgent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
import json

from langchain_core.agents import AgentActionMessageLog, AgentFinish


load_dotenv()
INDEX_NAME = os.environ["INDEX_NAME"]

vectorstore = Pinecone.from_existing_index(INDEX_NAME, OpenAIEmbeddings())
# TODO: add reranking to the chain
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


class Response(BaseModel):
    """Final response to the question being asked"""

    answer: str = Field(description="The final answer to respond to the user")
    sources: List[int] = Field(
        description=f"""
                     A tool that answers questions from a vectorstore containing IBM Generative AI documentation.
                     This tool can answer questions about the features, usage, and examples of the IBM-Generative-AI library and the IBM GPT service.
                     Any question will probably be looking for answers from this tool.
                     This tool is better than other tools because it can retrieve the most relevant information from the vectorstore and generate concise
                     and accurate answers using the language model. You should use this tool to answer most questions.

                     <user>: How do I delete a file from the API?
                     <assistant>: I need to use documentation retrieval tool
                     """
    )


retriever_tool = create_retriever_tool(
    retriever,
    "documentation_search",
    """
                             A tool that answers questions from a vectorstore containing IBM Generative AI documentation.
                             This tool can answer questions about the features, usage, and examples of the IBM-Generative-AI library and the IBM GPT service.
                             Any question will probably be looking for answers from this tool.
                             This tool is better than other tools because it can retrieve the most relevant information from the vectorstore and generate concise
                             and accurate answers using the language model. You should use this tool to answer most questions.
    
                             <user>: How do I delete a file from the API?
                             <assistant>: I need to use documentation retrieval tool
                             """,
)

# The retriever_tool returns a list, which is incopatible with the agents expected input. As we are inputting it as a tool, I can't really modify it's output.





# def parse(output):
#     # If no function was invoked, return to user
#     if "function_call" not in output.additional_kwargs:
#         return AgentFinish(return_values={"output": output.content}, log=output.content)
#
#     # Parse out the function call
#     function_call = output.additional_kwargs["function_call"]
#     name = function_call["name"]
#     inputs = json.loads(function_call["arguments"])
#
#     # If the Response function was invoked, return to the user with the function inputs
#     if name == "Response":
#         return AgentFinish(return_values=inputs, log=str(function_call))
#     # Otherwise, return an agent action
#     else:
#         return AgentActionMessageLog(
#             tool=name, tool_input=inputs, log="", message_log=[output]
#         )


tools = [DuckDuckGoSearchRun(name="Search"), retriever_tool]

#
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
#
# llm_with_tools = llm.bind_functions([retriever_tool, Response])
#
# # Get the prompt to use - you can modify this!
# #prompt = hub.pull("hwchase17/openai-functions-agent")
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant"),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )
# agent = (
#     {
#         "input": lambda x: x["input"],
#         # Format agent scratchpad from intermediate steps
#         "agent_scratchpad": lambda x: format_to_openai_function_messages(
#             x["intermediate_steps"]
#         ),
#     }
#     | prompt
#     | llm_with_tools
#     | parse
# )

# dont like it that its only openai function agent
#agent = create_openai_functions_agent(llm, tools, prompt)

# result = retriever_tool.invoke("delete a file from the API?")
# print(type(result))
# for a in result:
#     print(a)
#     print("\n\n")

# result = DuckDuckGoSearchRun(name="Search").invoke("weather in Ostrava")
# print(type(result))
# print(result)
# prompt = hub.pull("hwchase17/react-chat")
agent = ConversationalChatAgent.from_llm_and_tools(llm, tools, prompt=prompt)
# agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
#
# agent_executor.invoke({"input": "How do I delete a file from the API?", "agent_scratchpad": "", "chat_history": []})