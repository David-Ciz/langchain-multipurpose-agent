import os

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from langchain.chains import LLMChain, RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain.callbacks import StdOutCallbackHandler
from langchain.tools import BaseTool
from langchain_core.callbacks import StreamingStdOutCallbackHandler

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
INDEX_NAME = "documentation-chat"


def process_llm_response(llm_response):
    print(llm_response)
    answer_part = llm_response['answer']
    answer_part = answer_part + "\n\nSources:"
    #print(llm_response['result'])
    #print('\n\nSources:')
    for source in llm_response["source_documents"]:
        #print(source.metadata['source'])
        answer_part = answer_part + f"{source.metadata['source']} \n\n"
    return answer_part


handler = StdOutCallbackHandler()
llm_callbacks = [StreamingStdOutCallbackHandler()]

# Optionally, specify your own session_state key for storing messages
msgs = StreamlitChatMessageHistory(key="special_app_key")

memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, output_key='answer')
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

vectorstore = Pinecone.from_existing_index(INDEX_NAME, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

template = """You are an AI chatbot having a conversation with a human.
{summaries}
{chat_history}
Human: {question}
AI: """
prompt = PromptTemplate(input_variables=["summaries", "question"], template=template)
# Add the memory to an LLMChain as usual
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, callbacks=llm_callbacks)
chain_type_kwargs = {"prompt": prompt}
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever=retriever,
                                       memory=memory, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)


for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    response = qa_chain({"question": prompt}, callbacks=[handler])
    #response = qa_chain.call(question=prompt, human_input=prompt)
    processed_response = process_llm_response(response)
    #answer = response['answer']
    st.chat_message("ai").write(processed_response)