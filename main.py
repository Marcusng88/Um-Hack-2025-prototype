import streamlit as st
import time
# api call
from api import mex_api as api
from api import speech_to_text2 as speech
from api import deep_search

st.set_page_config(page_title="📝 Task Manager", layout="centered")

st.image("grab-logo.png", width=150)
st.title("Grab MEX Chatbot")

st.markdown(
    """
    <style>
    div[data-testid="stChatInput"] {
        background-color: #58BC6B; 
        padding: 3px; 
        border-radius: 300px; 
        box-shadow: 0px 8px 15px rgba(88, 188, 107, 0.8); 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
col1,col2 = st.columns([3,1])
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state["last_user_prompt"] = prompt  # Store for deep thinking
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("MEX Assistant is thinking..."):
            df = api.choose_dataset(prompt)
            api.query_pipeline(df)
            assistant_response,graph = api.mex_prompt(prompt)
            if graph is not None and not isinstance(graph, str):
                st.plotly_chart(graph)
            
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Two buttons side-by-side
col1, col2, col3, col4, col5 = st.columns([1.4, 1.6, 1, 1, 1])
with col1:
    deep_thinking = st.button("🤖 Deep Thinking")

with col2:
    transcription = st.button("🎙️ Start Transcription")

# Handle deep thinking and transcription
if deep_thinking and "last_user_prompt" in st.session_state:
    
    deep_prompt = st.session_state["last_user_prompt"]

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("Performing deep search..."):
            print(0)
            deep_response = deep_search.deep_search(deep_prompt)
            
            print(deep_response)  # Debugging

            for chunk in deep_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response) # Debugging

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.write("Deep thinking completed.")

elif transcription:
    st.write("👂🏻Listening... Speak into your microphone.")
    if prompt := speech.speech_prompt():
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("MEX Assistant is thinking..."):
                df = api.choose_dataset(prompt)
                api.query_pipeline(df)
                assistant_response,graph = api.mex_prompt(prompt)
            
                if graph is not None and not isinstance(graph, str):
                    st.plotly_chart(graph)

                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        prompt = ""
