import os


def env_variables_checker() -> str | None:
    # List of required env variables
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "INDEX_NAME"]
    # List of missing variables using list comprehension
    missing_vars = [var for var in required_vars if os.environ.get(var) is None]
    # If there are any missing variables, return a string error with their names
    if missing_vars:
        warning = "".join(missing_vars)
        return warning
    # Otherwise, return None
    else:
        return None