import os
import io
import gspread
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import openai
import base64

load_dotenv()  # Load environment variables from .env file

def authenticate_gdrive():
    """Authenticate and return a Google Drive client."""
    scope = ["https://www.googleapis.com/auth/drive"]
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    drive_service = build('drive', 'v3', credentials=creds)
    return drive_service

def list_jpg_files_in_directory(drive_service, folder_id):
    """List all JPEG files in a specified Google Drive directory."""
    query = f"'{folder_id}' in parents and mimeType='image/jpeg'"
    results = drive_service.files().list(q=query).execute()
    return results.get('files', [])

def download_file(drive_service, file_id, file_name):
    """Download a file from Google Drive."""
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.seek(0)
    with open(file_name, 'wb') as f:
        f.write(fh.read())

def upload_file(drive_service, folder_id, file_name, mime_type='text/plain'):
    """Upload a file to Google Drive."""
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_name, mimetype=mime_type)
    drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

def load_system_prompt(prompt_file_path):
    """Load the system prompt from a text file."""
    with open(prompt_file_path, 'r') as file:
        return file.read()

def generate_image_description(image_path, prompt_file_path):
    """Generate a description for an image using OpenAI."""
    openai.api_key = os.getenv('OPENAI_API_KEY')
    system_prompt = load_system_prompt(prompt_file_path)
    
    with open(image_path, 'rb') as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                ]
            }
        ],
        max_tokens=300
    )
    return response['choices'][0]['message']['content']

def process_images_in_directory(frames_folder_id, analysis_folder_id, prompt_file_path):
    """Process JPEG images in a Google Drive directory and save descriptions."""
    
    drive_service = authenticate_gdrive()
    jpg_files = list_jpg_files_in_directory(drive_service, frames_folder_id)
    
    for jpg_file in jpg_files:
        file_id = jpg_file['id']
        file_name = jpg_file['name']
        
        # Download the image
        download_file(drive_service, file_id, file_name)
        
        # Generate description
        description = generate_image_description(file_name, prompt_file_path)
        
        # Save description to a text file
        text_file_name = f"{os.path.splitext(file_name)[0]}.txt"
        with open(text_file_name, 'w') as text_file:
            text_file.write(description)
        
        # Upload the text file to the "frame_analysis" folder
        upload_file(drive_service, analysis_folder_id, text_file_name)
        
        # Clean up local files
        os.remove(file_name)
        os.remove(text_file_name)

if __name__ == "__main__":
    root_folder_id = input("Please enter root Google Drive folder ID: ")
    frames_folder_id = root_folder_id + "/frames"
    analysis_folder_id = root_folder_id + "/frame_analysis"
    prompt_file_path = 'system_prompt.txt'  # Path to the system prompt file
    process_images_in_directory(frames_folder_id, analysis_folder_id, prompt_file_path)