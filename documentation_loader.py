import os
import time
from langchain.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.utils.html import (PREFIXES_TO_IGNORE_REGEX,
                                  SUFFIXES_TO_IGNORE_REGEX)
from langchain.vectorstores import Pinecone
from bs4 import BeautifulSoup
import logging
import re
import pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
# load from env file the api key
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
INDEX_NAME = "documentation-chat"


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Documentation is contained within articles, all html elements outside it are noise we filter out.
    body = soup.find("article")
    # Could be useful to split it by headers section so the source would point to the actual part of the documentation.
    headers_to_split_on = [
        ("h1", "Header 1"),
    ]

    # Extra lines might not matter for some embedding algorithms, but shouldn't hurt either.
    return re.sub(r"\n\n+", "\n\n", body.text).strip()


def update_docs_database():
    docs_from_documentation = RecursiveUrlLoader(
        url="https://ibm.github.io/ibm-generative-ai/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://ibm.github.io/ibm-generative-ai/_static",
        ),
    ).load()
    logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs_transformed = text_splitter.split_documents(docs_from_documentation)
    logger.info(f"Split into {len(docs_transformed)} chunks from docs")
    # Uses "text-embedding-ada-002". Might exchange for something more powerful/cheaper/faster if needed
    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    # delete index if exists
    if INDEX_NAME in pinecone.list_indexes():
        pinecone.delete_index(INDEX_NAME)

    # create new index
    logger.info("Creating new Pinecone index")
    pinecone.create_index(
        name=INDEX_NAME,
        metric='cosine',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )

    # wait for index to be initialized
    while not pinecone.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

    index = pinecone.GRPCIndex(INDEX_NAME)
    logger.info(f"Pinecode index created with the following stats: {index.describe_index_stats()} ")
    logger.info("Upserting documents")
    try:
         doc_search = Pinecone.from_documents(docs_transformed, embeddings, index_name=INDEX_NAME)
    except Exception as e:
        # Might change to index.upsert if it proves unstable.
        print(f"Upsert operation failed: {e}")
    logger.info(f"Documents are being uploaded to the Pinecone database")


## Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Reconnect to the index by name
index = pinecone.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings()

# Create a vector store object
vectorstore = Pinecone(index, embeddings.embed_query, "text")
llm = OpenAI(temperature=0)
query = "How would I use the system to explain my code?"
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever,  return_source_documents=True)
answer = qa_chain(query)
process_llm_response(answer)
