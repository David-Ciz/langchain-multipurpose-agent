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
import click
# load from env file the api key
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
INDEX_NAME = "documentation-chat"


@click.group()
def cli():
    pass


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Documentation is contained within articles, all html elements outside it are noise we filter out.
    body = soup.find("article")
    # Extra lines might not matter for some embedding algorithms, but shouldn't hurt either.
    return re.sub(r"\n\n+", "\n\n", body.text).strip()


@cli.command("update")
def update_docs_database():
    """
    Scrapes the online documentation, extracts the relevant information, splits it into reasonable sized documents
    and saves it to a vector database. I used Langchain, Pinecone and OpenAI embeddings, but few line changes
    can make it work with any vector database or embeddings. Took some shameless inspiration
    from https://github.com/langchain-ai/chat-langchain/tree/master
    TODO: things/ ideas to improve this in the future:
    1: The splitting of documents is done just based on chunk sizes. Could be improved by using header sections and
       incorporating the header links for better sourcing of the data. I don't think RecursiveUrlLoader can be used for
       this, needs to be more manually processed with BF4, maybe some help can be gained from HTMLHeaderTextSplitter.
    2: Currently there is no checking if the documentation has changed, I think git actions that would fire when the
       documentation folder is committed into could rerun this method.
    3: I'm not really a fan of the forced async Pinecone.from_documents upserting and not getting a feedback if it's
       succeeded fully, feels like it could silently fail but I might be wrong here. For peace of mind might want to
       change it later to upserting through index.
    """
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
        Pinecone.from_documents(docs_transformed, embeddings, index_name=INDEX_NAME)
    except Exception as e:
        # Might change to index.upsert if it proves unstable.
        print(f"Upsert operation failed: {e}")
    logger.info(f"Documents are being uploaded to the Pinecone database")


def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


@cli.command()
def test_query():
    """ A quick test on a very basic question to see if the documentation is there."""
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
