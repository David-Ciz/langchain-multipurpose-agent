from abc import ABC

from langchain_core.callbacks import BaseCallbackHandler
import streamlit as st


class StreamHandler(BaseCallbackHandler, ABC):
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


def process_source_metadata(llm_response):
    """
    :param llm_response:
    :return:
    """
    answer_part = llm_response['answer']
    answer_part = answer_part + "\n\nSources: \n\n"
    for source in llm_response["source_documents"]:
        answer_part = answer_part + f"{source.metadata['source']} \n\n"
    return answer_part


def parse_agent_messages(msgs):
    # Due to the output of the agent with StreamlitCallbackHandler being more structured, this is required to make it nicer.
    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            # Render intermediate steps if any were saved
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                    st.write(step[0].log)
                    print(step[0].log)
            st.write(msg.content)

