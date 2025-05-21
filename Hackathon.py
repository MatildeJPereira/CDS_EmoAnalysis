import requests
import json
import time

API_KEY = "aETXkrhLOh1VIaauFwTw9gJCXnIgsmOg"
API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "mistral-small"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def chat_with_mistral(messages):
    """
    Sends a list of messages to the Mistral chat API and returns the assistant's response.

    Parameters:
        messages (list): A list of message dictionaries in the OpenAI chat format.
                         Example: [{"role": "user", "content": "Hello!"}]

    Returns:
        str: The assistant's reply as a string.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,     # Creativity level (0 = deterministic, 1 = more random)
        "top_p": 1.0,           # Nucleus sampling parameter
        "stream": False         # Disable streaming for simple usage
    }

    # Send a POST request to Mistral's API
    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))

    # Raise an error if the request failed
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} - {response.text}")

    # Parse the JSON response
    response_data = response.json()

    # Extract and return the assistant's reply
    return response_data['choices'][0]['message']['content']


def clean_response_json(response):
    split1, split2 = response.split("```json")
    split2_1, split2_2 = split2.split('```')
    if split2_1:
        json_data = json.loads(split2_1)
        if isinstance(json_data, dict) and len(json_data) == 1:
            return next(iter(json_data.values()))
        else:
            return json_data
    else:
        print("No JSON content found in the text.")
        return None


chat_history_eng = [
    {"role": "user",
     "content":"I want to make multiple prompts to an LLM with the same phrase in english and italian, and I want to use a json file. I need 70 phrases, each needs to ask for opinions. Can you make that json for me, with all the 70 phrases?"}
]
res = ""
try:
    res = chat_with_mistral(chat_history_eng)
    print(res)
except Exception as e:
    print("Error:", str(e))

json_res = clean_response_json(res)

for item in json_res:
    time.sleep(1)
    item["english_response"] = chat_with_mistral([{"role": "user", "content": item["english"]}])
    time.sleep(1)
    item["italian_response"] = chat_with_mistral([{"role": "user", "content": item["italian"]}])

print(json_res)
with open('static/json/mistral_response.json', 'w', encoding='utf-8') as f:
    json.dump(json_res, f)
