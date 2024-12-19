import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_texts(texts, num_words=10000, max_sequence_length=50):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_sequence_length)
    return padded_sequences, tokenizer 

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def load_system_prompt(prompt_file_path):
    """Load the system prompt from a text file."""
    with open(prompt_file_path, 'r') as file:
        return file.read()

def generate_frame_description(image_url, prompt_file_path):
    """Generate a description for a frame using OpenAI."""
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    system_prompt = load_system_prompt(prompt_file_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyse this Civ VI screenshot as instructed in your system prompt. Return the JSON only."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content