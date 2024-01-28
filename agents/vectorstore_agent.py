import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import ConversationalChatAgent, AgentExecutor, create_react_agent
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.tools import Tool
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from utils import process_source_metadata


class VectorStoreAgent:
    def __init__(self, vectorstore: Pinecone, llm: ChatOpenAI):
        self.vectorstore = vectorstore  # Initialize the vectorstore attribute
        self.llm = llm  # Initialize the llm attribute
        self.retriever = vectorstore.as_retriever(search_type="similarity",
                                                  search_kwargs={"k": 5})  # Initialize the retriever attribute
        self.template = """You are an AI chatbot having a conversation with a human. Use primarily the information from
the summaries. If you don't know the answer, it's better to say 'I'm not sure' rather than make stuff up. For the answer,
try to explain things you are returning and use large code examples to provide more context.
{summaries}
{chat_history}
Human: {question}
AI: """  # Initialize the template attribute
        self.prompt = ChatPromptTemplate.from_template(self.template)  # Initialize the prompt attribute
        self.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever=self.retriever,
                                                                    return_source_documents=True, )  # Initialize the qa_chain attribute

    def get_vectorstore_response(self, question):
        llm_response = self.qa_chain.invoke(question)  # Invoke the qa_chain with the question
        result = process_source_metadata(llm_response)  # Process the source metadata
        return result  # Return the result

    def as_tool(self):
        tool = Tool(
            name="Documentation retrieval",
            func=self.get_vectorstore_response,
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
        )
        return tool
