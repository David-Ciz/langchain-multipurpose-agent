import logging
import os

import click
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
INDEX_NAME = os.environ["INDEX_NAME"]


@click.group()
def cli():
    pass

# TODO: can't make it in time, the cli would need to call methods that are not bound to streamlit but fairly easy to add. Same for the api, just add fastapi decoration to the methods @app.post("/prompt")
# @cli.command()
# def call_documentation_qa_agent(prompt):
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, chat_memory=msgs,
#                                       output_key=memory_output_key)
#     executor = initiate_documentation_agent(INDEX_NAME, chat_box, memory)
#     get_documentation_qa_response(executor, chat_box, prompt)
#
# cli.command()
# def call_tools_agent(prompt):
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, chat_memory=msgs,
#                                       output_key=memory_output_key)
#     executor = initiate_documentation_agent(INDEX_NAME, chat_box, memory)
#     get_documentation_qa_response(executor, chat_box, prompt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()