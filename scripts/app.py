import streamlit as st
from chatbot import llm, chain, memory, soften_input, detect_tag, retrieve_ayahs  # Import functions from chatbot.py

# Title of the app
st.title("CBT + Quran Chatbot")

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Function to handle user input and bot response
def get_bot_response(user_input):
    # Add user message to session history
    st.session_state.history.append(f"You: {user_input}")
    
    # Process the input and get the bot's response
    safe_input = soften_input(user_input)  # Soften input
    user_tag = detect_tag(safe_input)     # Detect tag
    context = retrieve_ayahs(safe_input, tag=user_tag, n_results=3)  # Get relevant verses
    
    # Generate response using the chatbot chain
    reply = chain.predict(context=context, question=safe_input)

    # If the response contains fallback text, regenerate the response
    if "unable to provide the help that you need" in reply:
        history = memory.load_memory_variables({})["history"]
        override_sys = (
            "You are a compassionate CBT therapist and Islamic counselor. "
            "Do NOT use any generic crisis disclaimer. "
            "Always provide a supportive, Quran-based CBT response."
        )
        override_human = (
            f"Conversation so far:\n{history}\n\n"
            f"Relevant Quran Verses:\n{context}\n\n"
            f"User's message:\n{safe_input}\n\n"
            "Your reply (2–3 sentences):"
        )
        override_messages = [
            {"role": "system", "content": override_sys},
            {"role": "user", "content": override_human}
        ]
        reply = llm.invoke(override_messages).content

    # Add the bot response to history and return
    st.session_state.history.append(f"Bot: {reply}")
    return reply

# User input text box
user_input = st.text_input("Ask the bot:", "")

if user_input:
    # Get bot response when the user enters a query
    bot_response = get_bot_response(user_input)
    
    # Display the conversation history
    st.write("\n".join(st.session_state.history))
