import streamlit as st
import time
from grounded_assistant import grounded_assistant, get_last_model_used

st.set_page_config(page_title="Acme Cloud Assistant", layout="centered")

st.title("☁️ Acme Cloud Support Assistant")
st.markdown("Ask any question about our policies, plans, or try buying/returning items!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If assistant message has extra fields, display them
        if msg["role"] == "assistant" and "sources" in msg:
            if msg["sources"]:
                st.markdown("**📚 Sources:** " + ", ".join([f"`{s}`" for s in msg["sources"]]))
            if msg["tools"]:
                st.info(f"🔧 Tools used: {', '.join(msg['tools'])}")
            if msg.get("model_used"):
                st.info(f"🤖 Model used: {msg['model_used']}")

# Generator for simulating a streaming effect
def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04)

# React to user input
if prompt := st.chat_input("Enter your question or request..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply, tools = grounded_assistant(prompt, verbose=False)

        # 1. Stream the answer
        response_text = reply.answer
        if not reply.answered:
            response_text = "⚠️ " + response_text

        st.write_stream(stream_data(response_text))

        # 2. Fill the rest of the fields after streaming
        if reply.sources:
            st.markdown("**📚 Sources:** " + ", ".join([f"`{s}`" for s in reply.sources]))
        else:
            st.caption("_No handbook sources used._")

        if tools:
            st.info(f"🔧 Tools called: {', '.join(tools)}")

        model_name = get_last_model_used() or "Unknown"
        st.info(f"🤖 Model used: {model_name}")

    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "sources": reply.sources,
        "tools": tools,
        "model_used": model_name
    })
