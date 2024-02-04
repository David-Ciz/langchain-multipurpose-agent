import os

from dotenv import load_dotenv

from utils import env_variables_checker

MODEL_NAME = "gpt-3.5-turbo"
MODEL_NAME = "gpt-4"
TEMPERATURE = 0
# INDEX_NAME = os.environ["INDEX_NAME"]

def load_env() -> str:
    load_dotenv()
    warning = env_variables_checker()
    return warning
