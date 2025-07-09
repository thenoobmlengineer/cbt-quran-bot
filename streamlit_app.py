import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"
import streamlit as st
import time
import os
import sys

# Add the scripts directory to the path so we can import deploychat
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from deploychat import (
    soften_input,
    detect_tag,
    retrieve_ayahs,
    retrieve_hadiths,
    main_chain,
    followup_chain,
    closers,
    memory,
)

st.set_page_config(page_title="CBT + Quran & Hadith Chatbot", layout="wide")
st.title("CBT + Quran & Hadith Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

def get_response(user_input: str):
    # Prepare input
    safe = soften_input(user_input)
    tags = detect_tag(safe)
    ctx = retrieve_ayahs(safe, tags)
    hquery = tags[0] if tags else safe
    hctx = retrieve_hadiths(hquery)

    # Conditional context blocks
    q_block = f"Relevant Quran Verses:\n{ctx}\n\n" if ctx else ""
    h_block = f"Relevant Hadith:\n{hctx}\n\n" if hctx else ""

    # Main reply
    reply = main_chain.predict(
        history=memory.load_memory_variables({})["history"],
        q_block=q_block,
        h_block=h_block,
        question=safe
    )

    # Follow-up question
    followup = followup_chain.predict(user_input=user_input, bot_response=reply)

    return reply, followup

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    st.session_state.history.append(("You", user_input))
    bot_reply, bot_followup = get_response(user_input)
    st.session_state.history.append(("Bot", bot_reply))
    st.session_state.history.append(("Bot", bot_followup))

for speaker, message in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
    time.sleep(0.1)
