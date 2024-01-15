import os

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
INDEX_NAME = "documentation-chat"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


template = """ You are an expert programmer and problem-solver, tasked with answering any question
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end. Combine search results together into a coherent answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

vectorstore = Pinecone.from_existing_index(INDEX_NAME, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | custom_rag_prompt
#     | llm
#     | StrOutputParser()
# )
#

msgs = StreamlitChatMessageHistory(key="special_app_key")

memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)

qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, return_source_documents=True,
                                       memory=memory)


answer = qa_chain({"query": "How do I do Sampling substitution?", "chat_history": "Nothing"})

#answer = qa_chain("How do I do Sampling substitution?")
process_llm_response(answer)
#ConversationalRetrievalChain.from_llm()

#answer = rag_chain.invoke("How do I do Sampling substitution?")
#RunnableWithMessageHistory