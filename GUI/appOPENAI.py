# Import the necessary libraries
import streamlit as st
from openai import OpenAI

st.title("ENGINE - Amazon LLM <2GB Demo")

open_ai_key_file = "../apisaka.txt" # Your OPEN AI Key in this file
with open(open_ai_key_file, "r") as f:
    for line in f:
        openai_key = line
        break

client = OpenAI(api_key=openai_key)

# Check for existing session state variables
if "openai_model" not in st.session_state:
    # ... (initialize model)
    st.session_state.openai_model = "llama2-7b-Q4_K_M.gguf"
    
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_conversation():
    # ... (code for getting conversation history)
    if(len(st.session_state.messages) <= 10):
        return st.session_state.messages
    else:
        return st.session_state.messages[-10:]
    

def get_completion(prompt, model='gpt-4'):
    prompt_begin = "Complete the following conversation as the assistant: "
    mesg_hist = get_conversation()
    
    for i in range(len(mesg_hist)):
        prompt_begin += mesg_hist[i]['role'] + ": " + mesg_hist[i]['content']
        
    message_to_cli = {"role": "user", "content": prompt_begin + "U: " + prompt}
    response = client.chat.completions.create(
      model=model,
      messages=[message_to_cli]
    )
    return response.choices[0].message.content

for message in st.session_state.messages:
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

