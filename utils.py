import os
from dotenv import load_dotenv
from gdrive_utils import authenticate_gdrive, list_jpg_files, create_folder, upload_text_content
from openai_utils import generate_frame_description, generate_action_description
import base64
import io
from googleapiclient.http import MediaIoBaseDownload
import re
from openai import OpenAI

load_dotenv()  # Load environment variables from .env file

if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")

def ensure_gdrive_directories_exist(drive_service, root_folder_id, *directories):
    """Ensure that the specified directories exist in Google Drive, creating them if necessary."""
    folder_ids = {}
    
    for directory in directories:
        query = f"name='{directory}' and '{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        existing_folders = results.get('files', [])
        
        if existing_folders:
            folder_id = existing_folders[0]['id']
            print(f"Found existing directory: {directory} with ID: {folder_id}")
        else:
            folder_id = create_folder(drive_service, directory, root_folder_id)
            print(f"Created new directory: {directory} with ID: {folder_id}")
            
        folder_ids[directory] = folder_id
    
    return folder_ids

def get_image_as_base64(drive_service, file_id):
    """Download image from Drive and convert to base64."""
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    
    image_data = fh.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_image}"

def natural_sort_key(file):
    # Extract numbers from the filename using regex
    name = file['name']
    # This will extract all numbers from the filename
    numbers = [int(num) for num in re.findall(r'\d+', name)]
    return numbers  # Returns [1, 1] for frame_1_1, [1, 10] for frame_1_10

def process_frames(frames_folder_id, analysis_folder_id, actions_folder_id, frame_prompt_path, action_prompt_path):
    drive_service = authenticate_gdrive()
    jpg_files = list_jpg_files(drive_service, frames_folder_id)
    
    # Filter and sort the files
    jpg_files = [f for f in jpg_files if f['name'].endswith('.jpg')]
    jpg_files.sort(key=lambda x: int(os.path.splitext(x['name'])[0]))
    
    print(f"Processing {len(jpg_files)} frames...")

    # First, generate and upload all frame descriptions

    print("Generating frame descriptions...")
    for file in jpg_files:
        file_id = file['id']
        file_name = file['name']
        print(f"Processing frame: {file_name}")
        
        file_url = get_image_as_base64(drive_service, file_id)
        description = generate_frame_description(file_url, frame_prompt_path)
        
        text_file_name = f"{os.path.splitext(file_name)[0]}.json"
        print(f"Uploading description to frame_analysis folder")
        upload_text_content(drive_service, analysis_folder_id, text_file_name, description)

    # Then, generate action descriptions using only the screenshots
    print("Generating action descriptions...")
    for i in range(len(jpg_files) - 1):
        file_name1 = jpg_files[i]['name']
        file_name2 = jpg_files[i + 1]['name']
        file_id1 = jpg_files[i]['id']
        file_id2 = jpg_files[i + 1]['id']
        
        print(f"Generating action description for frames: {file_name1} and {file_name2}")
        file1_url = get_image_as_base64(drive_service, file_id1)
        file2_url = get_image_as_base64(drive_service, file_id2)
        
        action_description = generate_action_description(
            file1_url,
            file2_url,
            action_prompt_path
        )
        
        print(f"Uploading action description to actions_analysis folder")
        action_text_file_name = f"action_{os.path.splitext(file_name1)[0]}_{os.path.splitext(file_name2)[0]}.json"
        upload_text_content(drive_service, actions_folder_id, action_text_file_name, action_description)

    print("Game analysis complete.")

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

if __name__ == "__main__":
    root_folder_id = input("Please enter root Google Drive folder ID: ")
    drive_service = authenticate_gdrive()
    folder_ids = ensure_gdrive_directories_exist(drive_service, root_folder_id, 'frame_analysis', 'actions_analysis', 'frames')
    frames_folder_id = folder_ids['frames']
    analysis_folder_id = folder_ids['frame_analysis']
    actions_folder_id = folder_ids['actions_analysis']
    
    # Updated system prompt paths
    frame_prompt_path = 'system_prompts/frame_analysis_system_prompt.txt'
    action_prompt_path = 'system_prompts/action_analysis_system_prompt.txt'
    
    process_frames(frames_folder_id, analysis_folder_id, actions_folder_id, frame_prompt_path, action_prompt_path)