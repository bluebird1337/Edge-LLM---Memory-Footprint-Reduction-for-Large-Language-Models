import streamsync as ss
import requests

chat_history = ""

# This is a placeholder to get you started or refresh your memory.
# Delete it or adapt it as necessary.
# Documentation is available at https://streamsync.cloud
def get_completion(prompt, tokenNum=128, url='http://127.0.0.1:8080/completion'):    
    data = {
        "prompt": prompt,
        "n_predict": tokenNum,  # Set the maximum number of tokens to predict to 64
        "top_k": 10,  # Limit the next token selection to the 10 most probable tokens
        "temperature": 0.6  # Adjust the randomness of the generated text to 0.5 (lower values make the generation less random)
    }

    # Send the POST request
    response = requests.post(url, json=data)

    # Check the response
    if response.status_code == 200:
        # Request was successful, print the response
        return response.json()['content']
    return 0

def handle_message_simple(payload):
    chat_history += f"You: {payload}<split> BOB:"
    response = get_completion(payload)
    model_output = response.split("<split")

    chat_history += f"{model_output}<split>"
    return response