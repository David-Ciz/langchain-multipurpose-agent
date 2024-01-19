# ai-challenge solution
This repository contains a solution to the python ai-challenge.
## what you will find in this repository
This repository contains an application, a chat bot, that can answer questions from a documentation, with memory,
ability to handle large conversations and providing sources and streaming the answers. As a bonus, there is a secondary agent that can search the web
and execute python code. Other features to be implemented soon!

## Setup
in your preffered virtual environment use pip to install required packages from requirements.txt

```shell
# Create conda environment
$ conda create -n langchain python
$ conda activate langchain
$ pip install -r requirements.txt
```

## Running
# 1. create a .env file
We store enviromental variables in .env file located in a root directory. create this file with the following structure:
OPENAI_API_KEY=[YOUR_OPENAI_API_KEY]
PINECONE_API_KEY=[YOUR_PINECONE_API_KEY]
PINECONE_ENVIRONMENT=[YOUR_PINECONE_ENVIRONMENT_NAME]
INDEX_NAME = [YOUR_PINECONE_INDEX_NAME]
If you do not have these API keys, go to https://www.pinecone.io/ and https://platform.openai.com and obtain them (better instructions later)
# 2. Fill database
```shell
$ python documentation_loader.py update 
```
This will obtain the documentation pages from 'https://ibm.github.io/ibm-generative-ai/', preprocesses them and saves them into a pinecone database.
You can quickly test the functionality through the test-query method, that asks a basic question, you should see sources in the results from your documentation.
```shell
$ python documentation_loader.py test-query
```

# 3. Run the app
The app is created using streamlit, after running the following command a browser window should open with the app
```shell
$ streamlit run streamlit_app.py
```

# 4. Using the app
The chat interface is mostly self-explanatory, the important thing is the left panel, where you switch between the two agents.
The agents do share memory, which can be beneficial, but it also introduces instability and unexpected formatting. 
It is recommended to clear the memory when switching between the agents.


# 5. Planned features
- Better documentation extraction based on section headers
- adding the other  GenAI Node.js SDK documentation that I totally didn't miss in the instructions
- better document extraction with advanced techniques like reranking and better summarization
- better memory management with better chat history summarization
- cli/api interface
- answering questions from pdf/csv file
- dockerization
- type hints! I just rewrote stuff so much I didn't bother
- and more
