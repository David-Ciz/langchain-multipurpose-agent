# ai-challenge solution
This repository contains an application that serves to show my abilites for the python ai-challenge.
<meta> For why things are the way they are, I left good amount of meta comments and you can also read the thoughts.txt to get more insight on the code structure. </meta>
## what you will find in this repository
This repository contains an application, a chat bot, with memory,
ability to handle large conversations and providing sources and streaming the answers. It has access to tools that can search the web, answer questions from a documentation and execute python code. 
Other features to be implemented soon!

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
```
OPENAI_API_KEY=[YOUR_OPENAI_API_KEY]
PINECONE_API_KEY=[YOUR_PINECONE_API_KEY]
PINECONE_ENVIRONMENT=[YOUR_PINECONE_ENVIRONMENT_NAME]
INDEX_NAME = [YOUR_PINECONE_INDEX_NAME]
```
If you do not have these API keys, go to https://www.pinecone.io/ and https://platform.openai.com and obtain them (better instructions later if needed)
# 2. Fill database
```shell
$ python documentation_loader.py update 
```
This will obtain the documentation pages from 'https://ibm.github.io/ibm-generative-ai/', preprocesses them and saves them into a pinecone database.
You can quickly test the functionality through the test-query method, that asks a basic question, you should see sources in the results from your documentation.
```shell
$ python documentation_loader.py test-query
```
# 3 Running the app
We provide 3 options of interacting with the application, with main focus on streamlit frontend.
## 3.1 Run the app using streamlit
After running the following command a browser window should open with the app
```shell
$ streamlit run streamlit_app.py
```
The chat interface is mostly self-explanatory. After the first message a button appears for resetting the agent memory. 
Upload of the file while there, currently doesn't do much without the csv or pdf tool.

## 3.2 Run the app using cli
To use the cli interface, use the following command.
```shell
$ python cli_interface.py
```

## 3.3 Run the app using api
We also provide rudimetary api interface to interact with the agent. To start the server use:
```shell
$ python api_interface.py
```
To easily test the api, use 'quick_api_tests.py'

# 5. Planned features
- Better documentation extraction based on section headers and filtering out small documents
- adding the other  GenAI Node.js SDK documentation that I totally didn't miss in the instructions
- better document extraction with advanced techniques to better summarize 
- better memory management with better chat history summarization
- answering questions from pdf/csv file
- other llms support
- more databases support
- dockerization
- and more
