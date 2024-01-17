import os

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain.callbacks import StdOutCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
INDEX_NAME = "documentation-chat"


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")


def process_llm_response(llm_response):
    """
    :param llm_response:
    :return:
    """
    answer_part = llm_response['answer']
    answer_part = answer_part + "\n\nSources: \n\n"
    for source in llm_response["source_documents"]:
        answer_part = answer_part + f"{source.metadata['source']} \n\n"
    return answer_part


#handler = StdOutCallbackHandler()

# Optionally, specify your own session_state key for storing messages
msgs = StreamlitChatMessageHistory(key="special_app_key")

memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, output_key='answer')
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

vectorstore = Pinecone.from_existing_index(INDEX_NAME, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    with st.chat_message("ai"):
        chat_box = st.empty()
        stream_handler = StreamHandler(chat_box, display_method='write')

        template = """You are an AI chatbot having a conversation with a human.
        {summaries}
        {chat_history}
        Human: {question}
        AI: """
        prompt_template = PromptTemplate(input_variables=["summaries", "question"], template=template)
        # Add the memory to an LLMChain as usual
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, streaming=True, callbacks=[stream_handler])
        chain_type_kwargs = {"prompt": prompt_template}
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever=retriever,
                                                               memory=memory, return_source_documents=True,
                                                               chain_type_kwargs=chain_type_kwargs)
        full_response = ""
    # doesn't work with the callback , callbacks=[handler]
        for response in qa_chain.stream({"question": prompt}):
            #full_response += (response['answer'] or "")
            #chat_box.markdown(full_response + "▌")
        #chat_box.markdown(full_response)
    # #response = qa_chain.call(question=prompt, human_input=prompt)
            processed_response = process_llm_response(response)
            chat_box.write(processed_response)
    # #answer = response['answer']
