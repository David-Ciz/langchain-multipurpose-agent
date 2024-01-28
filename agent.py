import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import ConversationalChatAgent, AgentExecutor, create_react_agent, initialize_agent
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.tools import Tool
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from utils import process_source_metadata

load_dotenv()
INDEX_NAME = os.environ["INDEX_NAME"]


vectorstore = Pinecone.from_existing_index(INDEX_NAME, OpenAIEmbeddings())
# TODO: add reranking to the chain
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
query = "How would I use the system to explain my code?"
template = """You are an AI chatbot having a conversation with a human. Use primarily the information from
the summaries. If you don't know the answer, it's better to say 'I'm not sure' rather than make stuff up.
{summaries}
{chat_history}
Human: {question}
AI: """
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, streaming=True)
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever=retriever,
                                                       return_source_documents=True,)


def invoke_qa_chain(question):
    llm_response =qa_chain.invoke(question)
    result = process_source_metadata(llm_response)
    return result

# qa_chain = (
#             RunnableParallel({
#                 "context": itemgetter("question") | retriever,
#                 "question": itemgetter("question")
#             })
#             | {
#                 "response": prompt | llm,
#                 "context": itemgetter("context"),
#             }
#     )
# chain = prompt | llm
# qa_chain = (retriever)
#
# chain = RunnableParallel({
#                 "context": itemgetter("question") | retriever,
#                 "question": itemgetter("question")
#             }
#                          | {
#                              "response": prompt | llm,
#                              "context": itemgetter("context"),
#                          }
#                          )
# result = chain.invoke({"question":"Customize behavior of local client?"})
# print(result)



# result = invoke_qa_chain(qa_chain, "How do I delete a file from the API?")
# print(type(result))
# print(str(result))
# result = qa_chain.invoke({
#         "question": "shows how to create/retrieve/read/delete a file from API.",
# #     })
# from langchain import hub
# from langchain.agents import AgentExecutor, create_json_chat_agent, ConversationalChatAgent
#
# # prompt = hub.pull("hwchase17/react-chat-json")
# #
# documentation_retrieval = Tool(
#     name="documentation retrieval",
#     func=qa_chain.invoke,
#     description=f"""
#             A tool that answers questions from a vectorstore containing IBM Generative AI documentation.
#             This tool can answer questions about the features, usage, and examples of the IBM-Generative-AI library and the IBM GPT service.
#             Any question will probably be looking for answers from this tool.
#             This tool is better than other tools because it can retrieve the most relevant information from the vectorstore and generate concise
#             and accurate answers using the language model. You should use this tool to answer most questions.
#
#             <user>: How do I delete a file from the API?
#             <assistant>: I need to use documentation retrieval tool
#             """
# ),
# print(type(documentation_retrieval))
tools = [DuckDuckGoSearchRun(name="Search"),
             PythonREPLTool(),
         Tool(
             name="Documentation retrieval",
             func=invoke_qa_chain,
             description=f"""
             A tool that answers questions from a vectorstore containing IBM Generative AI documentation.
             This tool can answer questions about the features, usage, and examples of the IBM-Generative-AI library and the IBM GPT service.
             Any question will probably be looking for answers from this tool.
             This tool is better than other tools because it can retrieve the most relevant information from the vectorstore and generate concise
             and accurate answers using the language model. You should use this tool to answer most questions.
             When getting a result from this agent, pass it along to the user fully, with all the sources listed.
             Always also pass the sources.
             """,
             return_direct=True,
         ),
         ]
#prompt = hub.pull("hwchase17/react-chat")
template = """You are an AI chatbot having a conversation with a human. Use primarily the information from
    the summaries. If you don't know the answer, it's better to say 'I'm not sure' rather than make stuff up.
    {summaries}
    {chat_history}
    Human: {question}
    AI: """
prompt_template = PromptTemplate(input_variables=["summaries", "question"], template=template)
# #agent = create_json_chat_agent(llm, tools, prompt)
#
#
#
#
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, streaming=True, verbose=True, handle_parsing_errors=True,
                               return_intermediate_steps=True)

# # Using with chat history
# from langchain_core.messages import AIMessage, HumanMessage
# #
agent_executor.invoke(
    {
        "input": "How do I delete a file from the API? use documentation",
        "chat_history": []
    }
)
#


initialize_agent
# qa_chain.invoke("How do I delete a file from the API?")