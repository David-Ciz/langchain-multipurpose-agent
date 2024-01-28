import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


def process_source_metadata(llm_response: dict) -> str:
    """
    :param llm_response: llm dictionary response containing source documents metadata.
    :return: parsed answer
    """
    answer_part = llm_response['answer']
    answer_part = answer_part + "\n\nSources: \n\n"
    for source in llm_response["source_documents"]:
        answer_part = answer_part + f"{source.metadata['source']} \n\n"
    return answer_part


def parse_agent_messages(msgs: StreamlitChatMessageHistory):
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
