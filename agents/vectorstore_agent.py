from typing import Any

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore

from agents.output_parsers.parsers import process_source_metadata
from agents.prompts.prompt_templates import vectorstore_prompt


def create_vector_store_agent(vectorstore: VectorStore,
                              llm: BaseLanguageModel,
                              search_type: str = "similarity",
                              n_docs: int = 5,
                              chain_type: str = "stuff",
                              prompt_template: str = vectorstore_prompt(),
                              ):
    """
    Agent responsible for handling vectorstore information extraction and passing it to the user
    """
    retriever = vectorstore.as_retriever(search_type=search_type,
                                         search_kwargs={"k": n_docs})  # Initialize the retriever attribute

    prompt = ChatPromptTemplate.from_template(prompt_template)  # Initialize the prompt attribute
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,
                                                           chain_type=chain_type,
                                                           retriever=retriever,
                                                           return_source_documents=True,
                                                           chain_type_kwargs={
                                                               "verbose": True,
                                                               "prompt": prompt,
                                                           },

                                                           )
    return qa_chain


def get_vectorstore_response(vectorstore_agent: Runnable, prompt: dict[str, Any]) -> str:
    """
    Outputs a structured response with sources
    :param prompt: User or other agent question
    :return:
    """
    llm_response = vectorstore_agent.invoke(prompt)  # Invoke the qa_chain with the question
    result = process_source_metadata(llm_response)  # Process the source metadata
    return result
