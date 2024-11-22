from openai import OpenAI
import os
import io
from googleapiclient.http import MediaIoBaseDownload

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

def get_text_content(service, folder_id, filename):
    """Fetch text content from a file in Google Drive."""
    # Find the file by name in the specified folder
    query = f"name = '{filename}' and '{folder_id}' in parents"
    results = service.files().list(q=query, spaces='drive').execute()
    files = results.get('files', [])
    
    if not files:
        raise FileNotFoundError(f"No file found with name {filename}")
        
    # Get the content
    file_id = files[0]['id']
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    
    return fh.getvalue().decode('utf-8')

def generate_action_description(image_url1, image_url2, prompt_file_path):
    """Generate a description of the differences between two frames using OpenAI."""
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
                        "text": f"Analyse these two consecutive Civ VI screenshots according to the instructions in your system prompt. Return the JSON only."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url1
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url2
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content 