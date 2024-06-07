# Import the necessary libraries
import streamlit as st
import requests

url = 'http://127.0.0.1:7331/predict/'
st.title("ENGINE - Amazon LLM <2GB Demo - Our API")

# # TODO: Create or use client
# client = OpenAI(api_key=openai_key)

# # Check for existing session state variables
if "openai_model" not in st.session_state:
    # ... (initialize model)
    st.session_state.model = "ggml-model-Q4_K_M.gguf"
    
if "messages" not in st.session_state:
    st.session_state.messages = []
    
def get_conversation():
    # ... (code for getting conversation history)
    if(len(st.session_state.messages) <= 10):
        return st.session_state.messages
    else:
        return st.session_state.messages[-10:]
    
def get_completion(prompt):    
    data = {"prompt": prompt}
    response = requests.post(url, json=data)
    pred = response.json()['prediction']
    return pred

for message in get_conversation():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Wait for user input
if prompt := st.chat_input("What would you like to chat about?"):
    # ... (append user message to messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    # ... (display user message)
    st.chat_message("user").markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        # ... (send request to OpenAI API)
        assistant = get_completion(prompt)

        # ... (get AI response and display it)
        st.markdown(assistant)
        st.session_state.messages.append({"role": "assistant", "content": assistant})

    # ... (append AI response to messages)

