# LangChain Multipurpose Agent

This repository contains a versatile LangChain-based agent application with various capabilities for natural language processing and task execution.

## Features

This multipurpose agent includes the following capabilities:
- Persistent memory for context retention
- Handling of extended conversations
- Source attribution for information
- Real-time response streaming
- Integration with tools for web searching, documentation querying, and Python code execution

## Requirements

- Python 3.8 or higher

## Setup

In your preferred virtual environment, use pip to install the required packages from requirements.txt:

```shell
# Create conda environment
conda create -n langchain python
conda activate langchain
pip install -r requirements.txt
```

## Configuration

Create a .env file in the root directory with the following structure:

```
OPENAI_API_KEY=[YOUR_OPENAI_API_KEY]
PINECONE_API_KEY=[YOUR_PINECONE_API_KEY]
PINECONE_ENVIRONMENT=[YOUR_PINECONE_ENVIRONMENT_NAME]
INDEX_NAME=[YOUR_PINECONE_INDEX_NAME]
```

To obtain these API keys, visit https://www.pinecone.io/ and https://platform.openai.com.

## Database Initialization

To populate the Pinecone database with documentation:

```shell
python documentation_loader.py update
```

This command fetches documentation pages from 'https://ibm.github.io/ibm-generative-ai/', preprocesses them, and saves them into the Pinecone database.

To test the functionality, use:

```shell
python documentation_loader.py test-query
```

## Running the Application

The agent can be accessed through three different interfaces:

### 1. Streamlit Web Interface (Recommended)

```shell
streamlit run streamlit_app.py
```

This opens a browser window with the application. The chat interface includes a button for resetting the agent's memory.

### 2. Command-Line Interface

```shell
python cli_interface.py
```

### 3. API Interface

Start the server:

```shell
python api_interface.py
```

To test the API, use the 'quick_api_tests.py' script.

## Planned Enhancements

- Improved documentation extraction with section header analysis and small document filtering
- Advanced document summarization techniques
- Enhanced memory management with improved chat history summarization
- PDF and CSV file question answering capabilities
- Support for additional language models
- Expanded database integrations
- Docker containerization
- And more
