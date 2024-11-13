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
                        "text": "Analyse this Civ VI screenshot as instructed in your system prompt. Return the analysis as a text file."
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

def generate_action_description(drive_service, analysis_folder_id, image_url1, image_url2, file_name1, file_name2, prompt_file_path):
    """Generate a description of the differences between two frames using OpenAI."""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    system_prompt = load_system_prompt(prompt_file_path)
    
    # Get existing frame descriptions from Google Drive
    text_file_name1 = f"{os.path.splitext(file_name1)[0]}.txt"
    text_file_name2 = f"{os.path.splitext(file_name2)[0]}.txt"
    
    description1 = get_text_content(drive_service, analysis_folder_id, text_file_name1)
    description2 = get_text_content(drive_service, analysis_folder_id, text_file_name2)

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
                        "text": f"I am going to give you two screenshots and a text description of each..."  # your existing prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url1
                        }
                    },
                    {
                        "type": "text",
                        "text": description1
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url2
                        }
                    },
                    {
                        "type": "text",
                        "text": description2
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content 