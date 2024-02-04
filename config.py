from dotenv import load_dotenv
from utils import env_variables_checker

"""sets up basic configurations to easily change some llm basic parameters. Default for the agent is gpt-4 as
gpt 3.5-turbo even after many attempts at prompt engineering does not follow instructions for output formatting."""


# MODEL_NAME = "gpt-3.5-turbo"
MODEL_NAME = "gpt-4"
TEMPERATURE = 0
INDEX_NAME = "documentation-chat"
UPLOAD_DATA_PATH = "input_files/uploaded_data.csv"
def load_env() -> str:
    load_dotenv()
    warning = env_variables_checker()
    return warning