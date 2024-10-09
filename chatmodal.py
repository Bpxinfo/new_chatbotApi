import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
# Define the endpoint URL and key
endpoint_url = "https://Phi-3-small-128k-instruct-fnuun.eastus2.models.ai.azure.com/chat/completions"  # Replace with your endpoint URL
api_key = "q9IgR7TlzcFONONNB5KuIqwA2fIr21yP"  # Replace with your API key

# Example input data (you'll need to structure this according to your model's input)
# Request body (ensure this format is correct)
data = {
    "prompt": "Write a short story about a robot discovering emotions.",
    "max_tokens": 100,
    "temperature": 0.7,  # Adjust the randomness
    "stop": None  # Optional: you can specify a stopping condition, e.g. ["\n"] 
}

# Set headers, including the Authorization header with the API key
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Make the POST request
response = requests.post(endpoint_url, headers=headers, json=input_data)

# Check if the request was successful
if response.status_code == 200:
    # Parse and print the prediction result
    result = response.json()
    print("Prediction result:", result)
else:
    print(f"Error: {response.status_code}, {response.text}")
